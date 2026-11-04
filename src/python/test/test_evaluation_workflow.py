"""Known answers of ``starfinder.evaluation.workflow``, the engine comparison moved from ``scripts/workflow_golden.py``.

design/baselines/benchmark.md, R8: identity comparisons (``exact``, ``same_environment``) stay distinct from the
per-quantity numerical ones (``float64``: rtol 1e-9, atol 1e-12; ``float32``: rtol 1e-6, atol 1e-6). The retained
golden tests (``test_workflow_core_golden.py``, ``test_workflow_orchestration_golden.py``, ``test_workflow_cell_golden.py``)
call the comparison through the script.
"""
import hashlib

import numpy as np
import pandas as pd
import pytest

from starfinder.evaluation.workflow import (COMPARISON_KINDS, TOLERANCES, array_differences, record_differences,
                                            table_differences, value_differences, values_match)

from . import workflow_examples as X

pytestmark = [pytest.mark.evaluation, pytest.mark.contract]


def test_the_script_calls_the_package_comparison():
    golden = X.golden()
    assert golden.record_differences is record_differences and golden._close is values_match
    assert golden.TOLERANCES is TOLERANCES
    assert COMPARISON_KINDS == ("exact", "same_environment", "float64", "float32")
    assert TOLERANCES == {"float64": {"rtol": 1e-9, "atol": 1e-12}, "float32": {"rtol": 1e-6, "atol": 1e-6}}


@pytest.mark.parametrize("actual,expected,kind,match", [
    (3, 3, "exact", True), (3, 3.0, "exact", False), (True, 1, "exact", False), ("a", "a", "exact", True),
    # An identity is never relaxed to a tolerance, even for two numbers.
    (1.0, 1.0 + 1e-12, "exact", False), (1.0, 1.0 + 1e-12, "same_environment", False),
    ("sha256:ab", "sha256:ac", "same_environment", False),
    # |a - e| <= atol + rtol * |e|: 1e-12 + 1e-9 * 1 for float64, 1e-6 + 1e-6 * 1 for float32.
    (1.0 + 1.0e-9, 1.0, "float64", True), (1.0 + 1.5e-9, 1.0, "float64", False),
    (5e-13, 0.0, "float64", True), (2e-12, 0.0, "float64", False),
    (1.0 + 1.9e-6, 1.0, "float32", True), (1.0 + 2.1e-6, 1.0, "float32", False),
    (float("nan"), float("nan"), "float64", True), (float("inf"), float("inf"), "float64", True),
    # A Boolean or a string under a numerical class compares by identity.
    (True, 1.0, "float64", False), ("1.0", 1.0, "float64", False), (True, True, "float64", True),
])
def test_values_match_by_class(actual, expected, kind, match):
    assert values_match(actual, expected, kind) is match


def test_an_unknown_class_is_refused():
    with pytest.raises(ValueError, match="unknown comparison class"):
        values_match(1, 1, "approximate")


def test_record_differences_float_leaves_within_float64_and_every_other_leaf_exactly():
    expected = {"count": 2, "mean": 0.5, "status": "ok", "rows": [{"x": 1.0}, {"x": 2.0}], "gone": None}
    actual = {"count": 2, "mean": 0.5 + 4e-10, "status": "ok", "rows": [{"x": 1.0}, {"x": 2.0 + 1e-6}],
              "new": 1}
    assert record_differences(actual, expected, "r") == [
        "r.gone: only in expected", "r.new: only in actual", "r.rows[1].x: 2.000001 != 2.0 (float64)"]
    assert record_differences({"count": 2.0}, {"count": 2}) == [".count: 2.0 != 2 (exact)"]
    assert record_differences([1, 2], [1, 2, 3], "l") == ["l: 2 items != 3"]
    assert record_differences(expected, expected) == []


def test_value_differences_skip_identities_only_across_environments():
    expected = {"count": {"compare": "exact", "value": 7},
                "file:a.csv": {"compare": "same_environment", "value": "sha256:aa"},
                "fixture:b.tif": {"compare": "exact", "value": "sha256:bb"},
                "mean": {"compare": "float64", "value": 0.25},
                "only": {"compare": "exact", "value": 1}}
    actual = {"count": {"compare": "exact", "value": 7},
              "file:a.csv": {"compare": "same_environment", "value": "sha256:a0"},
              "fixture:b.tif": {"compare": "exact", "value": "sha256:b0"},
              "mean": {"compare": "float64", "value": 0.25 + 1e-13},
              "extra": {"compare": "float32", "value": 1.0}}
    same = value_differences(expected, actual, environment_keys=("fixture:",))
    assert same == ["extra: only in actual", "file:a.csv: 'sha256:a0' != 'sha256:aa' (same_environment)",
                    "fixture:b.tif: 'sha256:b0' != 'sha256:bb' (exact)", "only: only in expected"]
    other = value_differences(expected, actual, same_environment=False, environment_keys=("fixture:",))
    assert other == ["extra: only in actual", "only: only in expected"]
    # The numerical value is compared in either environment, by its class.
    actual["mean"]["value"] = 0.26
    assert "mean: 0.26 != 0.25 (float64)" in value_differences(expected, actual, same_environment=False)
    actual["count"]["compare"] = "float64"
    assert "count: comparison float64 != exact" in value_differences(expected, actual)


def test_table_differences_close_columns_and_integral_floats():
    expected = pd.DataFrame({"id": np.array([1, 2], dtype=np.int64), "z": [3.0, 4.0], "score": [0.1, 0.2],
                             "gene": pd.array(["a", None], dtype="string")})
    actual = expected.copy()
    actual["score"] = [0.1 + 1e-12, 0.2]
    # score holds non-integer floats: within the float64 tolerance by default; z holds integers: exactly.
    assert table_differences(actual, expected) == []
    actual["z"] = [3.0, 4.0 + 1e-12]
    assert table_differences(actual, expected) == ["z: 1 rows differ, first row 1: 4.000000000001 != 4.0"]
    actual["z"] = expected["z"]
    actual["gene"] = pd.array(["a", "b"], dtype="string")
    assert table_differences(actual, expected) == ["gene: 1 rows differ, first row 1: 'b' != None"]
    with pytest.raises(ValueError, match="non-integer floats"):
        table_differences(expected, expected, close=())
    assert table_differences(expected[["z", "id", "score", "gene"]], expected)[0].startswith("columns")
    assert table_differences(expected.astype({"id": "float64"}), expected)[0].startswith("dtypes")
    assert table_differences(expected.head(1), expected) == ["1 rows != 2"]


def test_array_differences_by_dtype():
    expected = {"labels": np.array([[1, 2]], dtype=np.uint32), "f32": np.array([1.0], dtype=np.float32),
                "f64": np.array([1.0, np.nan])}
    actual = {"labels": np.array([[1, 2]], dtype=np.uint32), "f32": np.array([1.0 + 1.5e-6], dtype=np.float32),
              "f64": np.array([1.0 + 5e-10, np.nan])}
    assert array_differences(expected, actual) == []
    actual["labels"] = np.array([[1, 3]], dtype=np.uint32)
    actual["f64"] = np.array([1.0 + 1e-8, np.nan])
    actual["f32"] = np.array([1.0], dtype=np.float64)
    actual["new"] = np.zeros(1)
    assert array_differences(expected, actual) == ["arrays f32: differ", "arrays f64: differ",
                                                   "arrays labels: differ", "arrays new: only in actual"]


def test_cross_engine_output_identity(tmp_path):
    """Two saved output inventories (relative path, bytes and SHA-256) of the same declared outputs match; one byte
    changed in one output is reported for that file only, in one environment, as an identity."""
    def inventory(root):
        return {f"file:{p.name}": {"compare": "same_environment",
                                   "value": f"{p.stat().st_size}:{hashlib.sha256(p.read_bytes()).hexdigest()}"}
                for p in sorted(root.iterdir())}
    for engine in ("python", "snakemake"):
        (tmp_path / engine).mkdir()
        (tmp_path / engine / "molecules.csv").write_bytes(b"z,y,x\n1,2,3\n")
        (tmp_path / engine / "run.json").write_bytes(b'{"status": "succeeded"}\n')
    assert value_differences(inventory(tmp_path / "python"), inventory(tmp_path / "snakemake")) == []
    (tmp_path / "snakemake" / "molecules.csv").write_bytes(b"z,y,x\n1,2,4\n")
    lines = value_differences(inventory(tmp_path / "python"), inventory(tmp_path / "snakemake"))
    assert len(lines) == 1 and lines[0].startswith("file:molecules.csv: ") and lines[0].endswith("(same_environment)")
    # Across environments a byte identity is not required: the quantities are compared by their classes instead.
    assert value_differences(inventory(tmp_path / "python"), inventory(tmp_path / "snakemake"),
                             same_environment=False) == []
