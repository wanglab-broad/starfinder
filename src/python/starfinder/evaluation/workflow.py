"""Pure comparison of two workflow runs' extracted values: identities exactly, quantities by their kind.

Moved from ``scripts/workflow_golden.py`` (R8), which still records and compares the workflow goldens with it. The
comparison classes are those of ``docs/contributing.md`` ("Expected values and tolerances"): ``exact`` (integers,
counts, strings, statuses, Booleans, integer images, file hashes of inputs), ``same_environment`` (the SHA-256 of an
output file: an identity within one environment, which a run in another environment need not reproduce),
``float64`` (rtol 1e-9, atol 1e-12) and ``float32`` (rtol 1e-6, atol 1e-6). An identity is never relaxed to a
numerical tolerance, and a numerical quantity is never compared by its bytes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["COMPARISON_KINDS", "TOLERANCES", "array_differences", "record_differences", "table_differences",
           "value_differences", "values_match"]

#: The tolerance of each numerical comparison class (``numpy.isclose``'s ``rtol`` and ``atol``).
TOLERANCES = {"float64": {"rtol": 1e-9, "atol": 1e-12}, "float32": {"rtol": 1e-6, "atol": 1e-6}}
#: Every comparison class, the two identities first and then the numerical classes of TOLERANCES.
COMPARISON_KINDS = ("exact", "same_environment", *TOLERANCES)


def values_match(actual, expected, kind) -> bool:
    """Whether two values match under one comparison class.

    Parameters
    ----------
    actual : object
        The value of the run being compared, a JSON value.
    expected : object
        The expected value, a JSON value.
    kind : str
        One of COMPARISON_KINDS.

    Returns
    -------
    bool
        ``exact`` and ``same_environment``, and any pair that is not two non-Boolean numbers: the same type and an
        equal value. ``float64`` and ``float32``: ``numpy.isclose`` with the class's tolerance, NaN equal to NaN.
    """
    if kind not in COMPARISON_KINDS:
        raise ValueError(f"unknown comparison class {kind!r}; choose from {list(COMPARISON_KINDS)}")
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (actual, expected))
    if kind in ("exact", "same_environment") or not numbers:
        return type(actual) is type(expected) and actual == expected
    return bool(np.isclose(actual, expected, equal_nan=True, **TOLERANCES[kind]))


def record_differences(actual, expected, where="") -> list[str]:
    """How a JSON record differs from an expected one, as readable lines (empty when it matches).

    Parameters
    ----------
    actual : object
        The record of the run being compared: a JSON value (mappings, lists and leaves), already without the fields
        that record where, when and by which revision a run was made.
    expected : object
        The expected record, in the same form.
    where : str
        The prefix of each line, the path of the record's root.

    Returns
    -------
    list of str
        One line per difference: a key only in one record, a list of another length, or a leaf that differs; a
        float leaf of ``expected`` compares within the ``float64`` tolerance, every other leaf exactly.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        lines = [f"{where}.{k}: only in {'expected' if k in expected else 'actual'}"
                 for k in sorted(set(expected) ^ set(actual))]
        for key in expected:
            if key in actual:
                lines += record_differences(actual[key], expected[key], f"{where}.{key}")
        return lines
    if isinstance(expected, list) and isinstance(actual, list):
        if len(actual) != len(expected):
            return [f"{where}: {len(actual)} items != {len(expected)}"]
        return [line for i, (a, e) in enumerate(zip(actual, expected))
                for line in record_differences(a, e, f"{where}[{i}]")]
    kind = "float64" if isinstance(expected, float) else "exact"
    return [] if values_match(actual, expected, kind) else [f"{where}: {actual!r} != {expected!r} ({kind})"]


def value_differences(expected, actual, *, same_environment=True, environment_keys=()) -> list[str]:
    """How two mappings of classified values differ, as readable lines (empty when they match).

    Parameters
    ----------
    expected : dict
        The expected values, ``{key: {"compare": kind, "value": value}}``, each kind one of COMPARISON_KINDS.
    actual : dict
        The values of the run being compared, in the same form.
    same_environment : bool
        Whether both were recorded in one environment. When False, the ``same_environment`` values and the keys
        that start with one of ``environment_keys`` are skipped.
    environment_keys : tuple of str
        Key prefixes of values that, like ``same_environment`` ones, hold only in the recording environment.

    Returns
    -------
    list of str
        A key only in one mapping, a key whose comparison class differs, or a value that does not match under the
        expected class.
    """
    lines = []
    for key in sorted(set(expected) | set(actual)):
        if key not in actual or key not in expected:
            lines.append(f"{key}: only in {'expected' if key in expected else 'actual'}")
            continue
        e, a = expected[key], actual[key]
        if e["compare"] != a["compare"]:
            lines.append(f"{key}: comparison {a['compare']} != {e['compare']}")
        elif not same_environment and (e["compare"] == "same_environment"
                                       or any(key.startswith(prefix) for prefix in environment_keys)):
            continue
        elif not values_match(a["value"], e["value"], e["compare"]):
            lines.append(f"{key}: {a['value']!r} != {e['value']!r} ({e['compare']})")
    return lines


def _cells(series):
    return [None if pd.isna(value) else value for value in series.astype(object)]


def _integral(values):
    finite = np.isfinite(values)
    return bool(np.array_equal(values[finite], np.round(values[finite])))


def table_differences(actual, expected, close=None, tolerance=TOLERANCES["float64"]) -> list[str]:
    """How a table differs from an expected one, as readable lines (empty when it matches).

    Parameters
    ----------
    actual : pandas.DataFrame
        The table of the run being compared.
    expected : pandas.DataFrame
        The expected table.
    close : iterable of str or None
        The columns of computed float values, compared with ``tolerance``. None: every float64 column of
        ``expected`` that holds a non-integer value; a float column of integer values compares exactly.
    tolerance : dict
        ``rtol`` and ``atol`` (default: the ``float64`` class).

    Returns
    -------
    list of str
        Column names and order, dtypes and the row count compare exactly, and stop the comparison when they
        differ; then one line per differing column with its count and first differing row. NaN equals NaN.

    Raises
    ------
    ValueError
        A column compared exactly holds non-integer floats in ``expected``.
    """
    if list(actual.columns) != list(expected.columns):
        return [f"columns {list(actual.columns)} != {list(expected.columns)}"]
    dtypes = {c: (str(actual[c].dtype), str(expected[c].dtype)) for c in expected
              if str(actual[c].dtype) != str(expected[c].dtype)}
    if dtypes:
        return [f"dtypes (actual, expected) {dtypes}"]
    if len(actual) != len(expected):
        return [f"{len(actual)} rows != {len(expected)}"]
    if close is None:
        close = [c for c in expected if str(expected[c].dtype) == "float64"
                 and not _integral(expected[c].to_numpy(np.float64))]
    differences = []
    for column in expected:
        if column in close:
            a = actual[column].to_numpy(np.float64)
            e = expected[column].to_numpy(np.float64)
            same = np.isclose(a, e, equal_nan=True, **tolerance)
        else:
            if expected[column].dtype.kind == "f" and not _integral(expected[column].to_numpy()):
                raise ValueError(f"column {column!r} holds non-integer floats: compare it with a tolerance")
            a, e = _cells(actual[column]), _cells(expected[column])
            same = np.array([x == y for x, y in zip(a, e)], dtype=bool)
        if not same.all():
            row = int(np.flatnonzero(~same)[0])
            differences.append(f"{column}: {int((~same).sum())} rows differ, first row {row}: "
                               f"{a[row]!r} != {e[row]!r}")
    return differences


def array_differences(expected, actual) -> list[str]:
    """How two mappings of named arrays differ, as readable lines (empty when they match).

    Parameters
    ----------
    expected : dict of str to numpy.ndarray
        The expected arrays by name.
    actual : dict of str to numpy.ndarray
        The arrays of the run being compared, by name.

    Returns
    -------
    list of str
        A name only in one mapping, or arrays that differ: dtype and shape exactly; a float32 array within the
        ``float32`` tolerance and another float array within the ``float64`` one (NaN equal to NaN); every other
        array element by element exactly.
    """
    lines = []
    for name in sorted(set(expected) | set(actual)):
        if name not in actual or name not in expected:
            lines.append(f"arrays {name}: only in {'expected' if name in expected else 'actual'}")
        elif expected[name].dtype != actual[name].dtype or expected[name].shape != actual[name].shape:
            lines.append(f"arrays {name}: differ")
        elif expected[name].dtype.kind == "f":
            tolerance = TOLERANCES["float32" if expected[name].dtype == np.float32 else "float64"]
            if not np.allclose(actual[name], expected[name], equal_nan=True, **tolerance):
                lines.append(f"arrays {name}: differ")
        elif not np.array_equal(expected[name], actual[name]):
            lines.append(f"arrays {name}: differ")
    return lines
