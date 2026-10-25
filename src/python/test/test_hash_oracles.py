"""The test-side oracles of hash_oracles.py against values written by hand from the documented rules (W-575, TM-18).

Each expected value is the documented encoding spelled out as bytes and hashed with hashlib, so an oracle that drifts
from its rule fails here; the oracles themselves stand in for the package's helpers in the verification tests.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from . import hash_oracles


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.dataset
def test_file_sha256_hashes_the_file_bytes(tmp_path):
    (tmp_path / "abc").write_bytes(b"abc")
    # The FIPS 180-2 test vector of "abc".
    assert hash_oracles.file_sha256(tmp_path / "abc") == \
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.stitching
def test_section_sha256_is_compact_sorted_utf8_json_and_refuses_non_finite_values():
    section = {"b": [1, 2.5], "a": "é", "c": None, "d": {"z": True, "y": "<NA>"}}
    text = b'{"a":"\xc3\xa9","b":[1,2.5],"c":null,"d":{"y":"<NA>","z":true}}'
    assert hash_oracles.section_text(section).encode("utf-8") == text
    assert hash_oracles.section_sha256(section) == sha256(text)
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            hash_oracles.section_sha256({"value": value})


@pytest.mark.stitching
def test_store_sha256_is_the_section_sha256_of_the_file_map(tmp_path):
    (tmp_path / "0" / "c").mkdir(parents=True)
    (tmp_path / "zarr.json").write_bytes(b"{}")
    (tmp_path / "0" / "c" / "0").write_bytes(b"\x01\x02")
    files = {"0/c/0": sha256(b"\x01\x02"), "zarr.json": sha256(b"{}")}
    assert hash_oracles.store_files(tmp_path) == files
    text = ('{"0/c/0":"' + files["0/c/0"] + '","zarr.json":"' + files["zarr.json"] + '"}').encode("utf-8")
    assert hash_oracles.store_sha256(tmp_path) == sha256(text)


@pytest.mark.segmentation
def test_record_sha256_is_default_json_with_ascii_escapes_nan_and_str_fallback():
    record = {"b": float("nan"), "a": "é", "c": Path("x/y"), "d": [1, float("inf"), -float("inf")], "e": None}
    text = b'{"a": "\\u00e9", "b": NaN, "c": "x/y", "d": [1, Infinity, -Infinity], "e": null}'
    assert hash_oracles.record_text(record).encode("utf-8") == text
    assert hash_oracles.record_sha256(record) == sha256(text)


@pytest.mark.stitching
def test_placement_table_sha256_hashes_the_escaped_csv_with_newline_line_ends():
    frame = pd.DataFrame({"fov_id": pd.array(["A", "<NA>", "\\x", None], dtype="string"),
                          "x": [0.1, 1.0, float("nan"), -float("inf")],
                          "n": pd.array([1, None, 3, 4], dtype="Int64"),
                          "kept": [True, False, True, False]})
    text = (b"fov_id,x,n,kept\n"
            b"A,0.10000000000000001,1,True\n"
            b"\\<NA>,1,<NA>,False\n"
            b"\\\\x,<NA>,3,True\n"
            b"<NA>,-inf,4,False\n")
    assert hash_oracles.placement_table_csv(frame) == text
    assert hash_oracles.placement_table_sha256(frame) == sha256(text)


@pytest.mark.segmentation
def test_array_sha256_prefixes_dtype_and_shape_to_native_c_order_bytes():
    array = np.arange(6, dtype="<u2").reshape(2, 3)
    expected = sha256(b"<u2|(2, 3)|" + bytes([0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0]))
    assert hash_oracles.array_sha256(array) == expected
    assert hash_oracles.array_sha256(array.astype(">u2")) == expected     # native byte order first
    assert hash_oracles.array_sha256(array.T) == sha256(b"<u2|(3, 2)|" + bytes([0, 0, 3, 0, 1, 0, 4, 0, 2, 0, 5, 0]))
    assert hash_oracles.array_sha256(np.array(7, dtype="<u1")) == sha256(b"|u1|(1,)|\x07")


@pytest.mark.synthetic
def test_bank_digest_hashes_name_dtype_shape_and_bytes_by_sorted_name():
    arrays = {"b": np.array([1.5], dtype="<f4"), "a": np.array(["x"], dtype="<U1")}
    data = b"a\0<U1\0(1,)\0" + b"x\0\0\0" + b"b\0<f4\0(1,)\0" + b"\x00\x00\xc0\x3f"
    assert hash_oracles.bank_digest(arrays) == sha256(data)


@pytest.mark.dataset
def test_read_table_unescapes_strings_and_reads_na_with_the_recorded_dtypes(tmp_path):
    path = tmp_path / "table.csv"
    path.write_bytes(b"name,count,score,flag,n,ok\n"
                     b"\\<NA>,1,0.5,True,1,True\n"
                     b"<NA>,<NA>,inf,<NA>,2,False\n"
                     b"\\\\x,2,<NA>,False,3,True\n"
                     b",3,-0.25,True,4,False\n")
    dtypes = {"name": "string", "count": "Int64", "score": "float64", "flag": "boolean", "n": "int64", "ok": "bool"}
    table = hash_oracles.read_table(path, dtypes)
    expected = pd.DataFrame({"name": pd.array(["<NA>", None, "\\x", ""], dtype="string"),
                             "count": pd.array([1, None, 2, 3], dtype="Int64"),
                             "score": [0.5, float("inf"), float("nan"), -0.25],
                             "flag": pd.array([True, None, False, True], dtype="boolean"),
                             "n": np.array([1, 2, 3, 4], dtype="int64"),
                             "ok": [True, False, True, False]})
    pd.testing.assert_frame_equal(table, expected, check_exact=True)
    assert table.name.iloc[0] == "<NA>" and pd.isna(table.name.iloc[1])
    with pytest.raises(ValueError, match="columns"):
        hash_oracles.read_table(path, dict(reversed(list(dtypes.items()))))
