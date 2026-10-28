"""Test-side oracles of the recorded hashes and of the CSV table format, written from the documented rules.

A test that verifies a recorded hash or reads a saved table compares with these functions rather than with the
package's own helper, so a defect in that helper cannot pass its own test (W-575, TM-18). Each function implements
one rule as the docs state it:

* ``file_sha256``: the SHA-256 of a file's bytes;
* ``section_sha256``: an ``export.json`` section (design/specs/sample-output.md, "Read-back checks", R1);
* ``store_files`` and ``store_sha256``: an OME-Zarr or SpatialData store's file map and its SHA-256
  (design/specs/sample-output.md, "The fusion section of `export.json`");
* ``record_sha256``: the ``record_sha256`` of a record given in memory (design/specs/assignment.md, "Run record");
* ``placement_table_csv`` and ``placement_table_sha256``: a placement table's CSV bytes and their SHA-256
  (design/specs/stitching.md, "Persistence");
* ``array_sha256``: the hash of a label or input array (design/specs/segmentation.md, "Run record");
* ``bank_digest``: an appearance bank's digest (docs/synthetic-real-appearance.md, "The appearance bank");
* ``read_table``: a CSV or Parquet table read back with its recorded dtypes (docs/checkpoints.md, "Table formats").

The module uses only hashlib, json and pandas, and imports nothing from starfinder.
"""
import hashlib
import json

import pandas as pd

NA_TOKEN = "<NA>"
ESCAPE = "\\"
STRING_DTYPES = ("string", "str")
# A recorded dtype -> the nullable dtype its CSV text is parsed as, before the cast to the recorded dtype.
PARSED_AS = {"int64": "Int64", "bool": "boolean", "str": "string"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path) -> str:
    """The SHA-256 of a file's bytes."""
    with open(path, "rb") as handle:
        return sha256(handle.read())


def section_text(section) -> str:
    """A section's canonical JSON: sorted keys, separators ``,`` and ``:`` (no spaces), non-ASCII characters
    written as themselves, and no NaN or infinity (``ValueError``)."""
    return json.dumps(section, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def section_sha256(section) -> str:
    """The SHA-256 of a section's canonical JSON, encoded as UTF-8."""
    return sha256(section_text(section).encode("utf-8"))


def store_files(folder) -> dict:
    """Every file under a store folder (a ``pathlib.Path``): its path relative to the folder, with ``/`` separators,
    to its SHA-256."""
    return {p.relative_to(folder).as_posix(): file_sha256(p) for p in folder.rglob("*") if p.is_file()}


def store_sha256(folder) -> str:
    """A store's ``sha256``: the section SHA-256 of its file map."""
    return section_sha256(store_files(folder))


def record_text(value) -> str:
    """The JSON text ``record_sha256`` hashes: sorted keys, separators ``, `` and ``: ``, non-ASCII characters
    escaped as ``\\uXXXX``, NaN and infinity written as ``NaN``, ``Infinity`` and ``-Infinity``, and any other
    object as its ``str``."""
    return json.dumps(value, sort_keys=True, default=str)


def record_sha256(value) -> str:
    """The SHA-256 of a record's JSON text (``record_text``), encoded as UTF-8."""
    return sha256(record_text(value).encode("utf-8"))


def _escaped(column):
    literal = column.eq(NA_TOKEN).fillna(False) | column.str.startswith(ESCAPE).fillna(False)
    return column.mask(literal, ESCAPE + column)


def placement_table_csv(frame) -> bytes:
    """The CSV bytes of a placement table: the table format of the checkpoints (a header row, no index, floats as
    ``%.17g``, a missing value as ``<NA>``, a string equal to ``<NA>`` or starting with ``\\`` with one more
    leading ``\\``), lines ending in ``\\n``, encoded as UTF-8."""
    text = frame.copy()
    for column in frame.columns:
        if str(frame[column].dtype) in STRING_DTYPES:
            text[column] = _escaped(frame[column].astype("string"))
    return text.to_csv(index=False, float_format="%.17g", na_rep=NA_TOKEN, lineterminator="\n").encode("utf-8")


def placement_table_sha256(frame) -> str:
    """The SHA-256 a placement record and an approval hold for a table."""
    return sha256(placement_table_csv(frame))


def _at_least_1d(array):
    return array.reshape(1) if array.ndim == 0 else array


def _native(array):
    return array if array.dtype.isnative else array.astype(array.dtype.newbyteorder("="))


def array_sha256(array) -> str:
    """The SHA-256 of the text ``"<dtype.str>|<shape>|"`` (UTF-8) followed by the array's C-order bytes, both in
    native byte order; the shape as Python writes a tuple, ``(4, 5)`` or ``(3,)``, an array with no axis as
    ``(1,)``."""
    array = _native(_at_least_1d(array))
    return sha256(f"{array.dtype.str}|{array.shape}|".encode("utf-8") + array.tobytes(order="C"))


def bank_digest(arrays) -> str:
    """The digest of a bank's arrays: for each array by sorted name, the text ``"<name>\\0<dtype.str>\\0<shape>\\0"``
    (UTF-8) followed by its C-order bytes in the array's own byte order, all hashed as one SHA-256; the shape as
    ``array_sha256`` writes it."""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = _at_least_1d(arrays[key])
        digest.update(f"{key}\0{value.dtype.str}\0{value.shape}\0".encode("utf-8"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _unescaped(column):
    missing = column.eq(NA_TOKEN)
    column = column.mask(column.str.startswith(ESCAPE), column.str.slice(len(ESCAPE)))
    return column.mask(missing, pd.NA)


def read_table(path, dtypes) -> pd.DataFrame:
    """A saved table read with the dtypes its record lists, in their order.

    CSV: a string column is read verbatim, ``<NA>`` read as missing and one leading ``\\`` removed; in any other
    column ``<NA>`` is missing; floats are read to the nearest float64; integers and booleans are read as their
    nullable dtypes, then every column is cast to the recorded dtype. Parquet: read, then cast the same way.
    """
    if str(path).endswith(".parquet"):
        frame = pd.read_parquet(path)
    else:
        strings = [c for c, d in dtypes.items() if d in STRING_DTYPES]
        frame = pd.read_csv(path, dtype={c: PARSED_AS.get(d, d) for c, d in dtypes.items()}, keep_default_na=False,
                            na_values={c: [NA_TOKEN] for c in dtypes if c not in strings},
                            float_precision="round_trip")
        for column in strings:
            frame[column] = _unescaped(frame[column])
    if list(frame.columns) != list(dtypes):
        raise ValueError(f"{path} has the columns {list(frame.columns)}, the record {list(dtypes)}")
    return frame.astype(dtypes)
