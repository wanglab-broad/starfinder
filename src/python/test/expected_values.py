"""Explicit expected values of the golden tests: the files under ``data/expected/`` and their comparison.

docs/contributing.md, "Expected values and tolerances" (W-526 TM-20). A table is a CSV file in the
checkpoint CSV convention: floats in ``%.17g`` (which round-trips float64), a missing value written as
``<NA>``, and a string equal to ``<NA>`` or starting with a backslash written with one leading
backslash. The dtype of every column is recorded in the folder's ``dtypes.json``
(``{file name: {column: dtype}}``). Arrays are ``.npz`` files. The files are read here with NumPy and
pandas only, never with package code, so the same reader also reads a checkpoint CSV that a test
wrote, with the dtypes of its header.

``table_differences`` compares column names, order and dtypes and the row count exactly. Columns
named in ``close`` (computed float64 values) compare with ``tolerance``, as
``numpy.testing.assert_allclose`` does (``|actual - expected| <= atol + rtol * |expected|``; NaN equals
NaN, an infinity only itself). Every other column compares exactly; a float column compared exactly
must hold integer values in the expectation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# docs/contributing.md: float64 computed values.
FLOAT64 = {"rtol": 1e-9, "atol": 1e-12}
NA = "<NA>"
ESCAPE = "\\"
DTYPES = ("string", "float64", "int64", "bool")


def write_csv(frame, path):
    """Write ``frame`` as an expected-value CSV; return its dtype map (for ``dtypes.json``)."""
    dtypes = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
    unsupported = {c: d for c, d in dtypes.items() if d not in DTYPES}
    if unsupported:
        raise ValueError(f"unsupported dtypes {unsupported}")
    text = frame.copy()
    for column, dtype in dtypes.items():
        if dtype == "string":
            strings = frame[column]
            literal = strings.eq(NA).fillna(False) | strings.str.startswith(ESCAPE).fillna(False)
            text[column] = strings.mask(literal, ESCAPE + strings)
    text.to_csv(path, index=False, float_format="%.17g", na_rep=NA)
    return dtypes


def _parse(cells, dtype):
    if dtype == "string":
        return pd.array([pd.NA if c == NA else c[len(ESCAPE):] if c.startswith(ESCAPE) else c for c in cells],
                        dtype="string")
    if dtype == "float64":
        return np.array([np.nan if c == NA else float(c) for c in cells], dtype=np.float64)
    if dtype == "int64":
        return np.array([int(c) for c in cells], dtype=np.int64)
    if dtype == "bool":
        return np.array([{"True": True, "False": False}[c] for c in cells], dtype=bool)
    raise ValueError(f"unsupported dtype {dtype!r}")


def read_csv(path, dtypes):
    """Read a CSV in the convention above with the given ``{column: dtype}``, in that column order."""
    text = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    if list(text.columns) != list(dtypes):
        raise ValueError(f"{Path(path).name}: columns {list(text.columns)} differ from {list(dtypes)}")
    return pd.DataFrame({column: _parse(text[column].tolist(), dtype) for column, dtype in dtypes.items()},
                        index=pd.RangeIndex(len(text)))


def read_table(path):
    """Read an expected table with the dtypes recorded for it in its folder's ``dtypes.json``."""
    path = Path(path)
    return read_csv(path, json.loads((path.parent / "dtypes.json").read_text())[path.name])


def read_arrays(path):
    """The arrays of an expected ``.npz`` file, by name."""
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def _cells(series):
    return [None if pd.isna(value) else value for value in series.astype(object)]


def table_differences(actual, expected, close=(), tolerance=FLOAT64):
    """How ``actual`` differs from ``expected`` (empty when it matches), as readable lines."""
    if list(actual.columns) != list(expected.columns):
        return [f"columns {list(actual.columns)} != {list(expected.columns)}"]
    dtypes = {c: (str(actual[c].dtype), str(expected[c].dtype)) for c in expected
              if str(actual[c].dtype) != str(expected[c].dtype)}
    if dtypes:
        return [f"dtypes (actual, expected) {dtypes}"]
    if len(actual) != len(expected):
        return [f"{len(actual)} rows != {len(expected)}"]
    differences = []
    for column in expected:
        if column in close:
            a = actual[column].to_numpy(np.float64)
            e = expected[column].to_numpy(np.float64)
            same = np.isclose(a, e, equal_nan=True, **tolerance)
        else:
            if expected[column].dtype.kind == "f":
                e = expected[column].to_numpy()
                finite = np.isfinite(e)
                if not np.array_equal(e[finite], np.round(e[finite])):
                    raise ValueError(f"column {column!r} holds non-integer floats: compare it with a tolerance")
            a, e = _cells(actual[column]), _cells(expected[column])
            same = np.array([x == y for x, y in zip(a, e)], dtype=bool)
        if not same.all():
            row = int(np.flatnonzero(~same)[0])
            differences.append(f"{column}: {int((~same).sum())} rows differ, first row {row}: "
                               f"{a[row]!r} != {e[row]!r}")
    return differences


def assert_table_matches(actual, expected, close=(), tolerance=FLOAT64, name=""):
    """Assert that ``table_differences`` finds no difference."""
    differences = table_differences(actual, expected, close, tolerance)
    assert not differences, f"{name}: " + "; ".join(differences)
