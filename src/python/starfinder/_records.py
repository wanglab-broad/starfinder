"""Record helpers the stages share: file and JSON hashes, JSON records, atomic writes, the checkpoint CSV codec
and the provenance of a record.

Private to the package and not exported. A hash that covers different bytes (an array, a table's CSV
text, a canonical section) keeps its own function next to the record that holds it.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import uuid

import numpy as np
import pandas as pd

# The written form of a missing wavelength (records and summaries).
UNAVAILABLE = "unavailable"


# --- Hashes ---------------------------------------------------------------------

def file_sha256(path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def check_file(path, recorded, what="file") -> None:
    """A recorded file must exist and have the recorded SHA-256; ValueError names the path and both hashes."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{what} {path} is missing (recorded SHA-256 {recorded})")
    actual = file_sha256(path)
    if actual != recorded:
        raise ValueError(f"{what} {path} has SHA-256 {actual}, the record {recorded}")


def json_sha256(value) -> str:
    """SHA-256 of a value's JSON text with sorted keys (other objects as their str)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


# --- JSON records -----------------------------------------------------------------

def jsonable(value):
    """Plain JSON values; tuples become lists, types qualified names, NaN/inf null."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def tuples(value):
    """Invert JSON list conversion for results whose sequences are tuples."""
    if isinstance(value, list):
        return tuple(tuples(v) for v in value)
    if isinstance(value, dict):
        return {k: tuples(v) for k, v in value.items()}
    return value


def json_record(record):
    """The record as it reads back from its JSON file (tuples become lists, non-finite floats null)."""
    return json.loads(json.dumps(jsonable(record), allow_nan=False))


@contextmanager
def atomic_file(path):
    """Yield a temporary sibling path; replace ``path`` with it only on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json(data, path):
    """Atomically write indented, strict JSON (non-finite floats become null)."""
    with atomic_file(path) as tmp:
        tmp.write_text(json.dumps(jsonable(data), indent=2, allow_nan=False) + "\n")


# --- Folder publication -------------------------------------------------------------

def atomic_path(path) -> Path:
    """A temporary sibling name for a file or folder that is renamed into place when complete."""
    path = Path(path)
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def publish(tmp, path) -> None:
    """Rename a completed temporary file or folder into place, replacing an earlier one."""
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    os.replace(tmp, path)


# --- The checkpoint table codec -------------------------------------------------------

TABLE_FORMATS = ("csv", "parquet")
NA_TOKEN = "<NA>"
ESCAPE = "\\"
_CONTROL = "[\x00-\x1f\x7f]"
# In-memory dtype -> dtype used to parse CSV text; the header restores the former.
_CSV_DTYPES = {"string": "string", "str": "string", "float64": "float64",
               "int64": "Int64", "Int64": "Int64", "bool": "boolean", "boolean": "boolean",
               "UInt32": "UInt32"}


def _require_parquet():
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise ImportError("Parquet checkpoints require pyarrow; install the 'checkpoint' extra "
                          "or use table_format='csv'") from error


def _dtype_map(frame):
    dtypes = {}
    for column, dtype in frame.dtypes.items():
        name = str(dtype)
        if name not in _CSV_DTYPES:
            raise ValueError(f"checkpoint column {column!r} has unsupported dtype {name}")
        dtypes[str(column)] = name
    return dtypes


def _escape_strings(column):
    """CSV text for a string column: missing values stay NA (written as NA_TOKEN).

    Literal values equal to NA_TOKEN or starting with ESCAPE gain one leading
    ESCAPE, so printable text, including "", "NA" or "<NA>", reads back exactly.
    """
    literal = column.eq(NA_TOKEN).fillna(False) | column.str.startswith(ESCAPE).fillna(False)
    return column.mask(literal, ESCAPE + column)


def _unescape_strings(column):
    missing = column.eq(NA_TOKEN)
    column = column.mask(column.str.startswith(ESCAPE), column.str.slice(len(ESCAPE)))
    return column.mask(missing, pd.NA)


def _write_table(frame, directory, name, table_format):
    """Write a table with its dtype map; return (file name, dtype map)."""
    dtypes = _dtype_map(frame)
    path = Path(directory) / f"{name}.{table_format}"
    if table_format == "parquet":
        _require_parquet()
        with atomic_file(path) as tmp:
            frame.to_parquet(tmp, index=False, engine="pyarrow")
    elif table_format == "csv":
        text = frame.copy()
        for column, dtype in dtypes.items():
            if dtype in ("string", "str"):
                strings = frame[column].astype("string")
                # CSV holds printable text only; never write a changed value.
                if strings.str.contains(_CONTROL, regex=True, na=False).any():
                    raise ValueError(f"CSV checkpoint column {column!r} contains a control character "
                                     "(U+0000-U+001F or U+007F); use table_format='parquet' for such data")
                text[column] = _escape_strings(strings)
        with atomic_file(path) as tmp:
            text.to_csv(tmp, index=False, float_format="%.17g", na_rep=NA_TOKEN)
    else:
        raise ValueError(f"table_format must be one of {TABLE_FORMATS}")
    return path.name, dtypes


def _read_table(path, dtypes):
    """Read a table and restore the recorded dtype of every column, in order."""
    path = Path(path)
    if path.suffix == ".parquet":
        _require_parquet()
        frame = pd.read_parquet(path, engine="pyarrow")
    else:
        # String columns are read verbatim (no NA parsing), then unescaped.
        strings = [c for c, d in dtypes.items() if d in ("string", "str")]
        frame = pd.read_csv(path, dtype={c: _CSV_DTYPES[d] for c, d in dtypes.items()},
                            keep_default_na=False, float_precision="round_trip",
                            na_values={c: [NA_TOKEN] for c in dtypes if c not in strings})
        for column in strings:
            if column in frame:
                frame[column] = _unescape_strings(frame[column])
    if list(frame.columns) != list(dtypes):
        raise ValueError(f"{path.name}: columns differ from the checkpoint header")
    return frame.astype(dtypes).reset_index(drop=True)


# --- Provenance -----------------------------------------------------------------------

_PACKAGES = ("starfinder", "numpy", "pandas", "scipy", "scikit-image", "tifffile", "pyarrow", "SimpleITK")


@lru_cache(maxsize=1)
def code():
    from importlib.metadata import PackageNotFoundError, version
    try:
        package_version = version("starfinder")
    except PackageNotFoundError:
        package_version = None
    root = Path(__file__).resolve().parent
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
        # An installed package can sit inside another project's repository; only
        # a starfinder checkout (src/python/starfinder at its root) is recorded.
        if not (Path(top) / "src" / "python" / "starfinder").resolve() == root:
            raise subprocess.SubprocessError("package is not inside a starfinder checkout")
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, timeout=30, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit = dirty = None
    return {"version": package_version, "git_commit": commit or None, "git_dirty": dirty}


@lru_cache(maxsize=1)
def environment():
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for name in _PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def software():
    return {"code": code(), "environment": environment()}
