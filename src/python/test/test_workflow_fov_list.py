"""The FOV list named in the configuration, so that a raw dataset folder is read in place (W-872).

The Python-only key ``fov_list`` names the FOV list (a ``fovs.csv``), absolute or relative to the input root; absent
or null, the plan reads ``<input_root>/fovs.csv`` as before (W-566). In a round's folder the plan lists the one file
of each configured channel label (``*<label>*.tif`` or ``.tiff``); the microscope's other files are not read, and a
label that matches two files is refused naming both. A regular grid places a sample's FOVs in the list's row order, so
the plan warns when the rows are in lexical order and a numeric FOV pattern gives another order. Fixtures, all written
in the test: an input root in the raw tissue-2D layout (Leica file names with spaces, an ``.hgsb`` and a
``_history.txt`` beside the channel files), its raw images empty files (the plan reads no image content), and the FOV
list in a folder of its own outside the input root.
"""
import copy
import json
import warnings
from pathlib import Path

import pytest

from starfinder.dataset.workflow import _fov_list, _lexical_order, _membership
from starfinder.workflow._plan import PlanRefused

from . import workflow_examples as X
from . import workflow_tasks as W

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

GRID = {"method": "multiview_stitcher", "input": "reference_merged",
        "positions": {"source": "regular_grid",
                      "regular_grid": {"columns": 2, "rows": 2, "order": "column_major_down_right", "overlap": 0.1}}}
NUMERIC = [f"tile_{i}" for i in range(1, 13)]
LEXICAL = sorted(NUMERIC)                     # tile_1, tile_10, tile_11, tile_12, tile_2, ...
#: The files of one round folder of the raw layout besides the channel files: neither matches a channel label.
EXTRA = ("{stem}.hgsb", "{stem}_history.txt")


def fov_rows(fovs, sample="tissue-2D"):
    return "sample,fov_id\n" + "".join(f"{sample},{fov}\n" for fov in fovs)


def raw_inputs(config, fovs):
    """The raw layout under the configuration's input root, with no fovs.csv: round<r>/<fov>/ holding
    ``9919seqA<r>_TileScan_001_Tile <n>_cmle_<label>.tif`` for each channel, an .hgsb and a _history.txt, and the
    codebook genes.csv. Returns the input root."""
    root = W.input_root(config)
    root.mkdir(parents=True)
    (root / "genes.csv").write_text("gene,barcode\n")
    for r in range(1, config["n_rounds"] + 1):
        for fov in fovs:
            folder = root / f"round{r}" / fov
            folder.mkdir(parents=True)
            stem = f"9919seqA{r}_TileScan_001_Tile {fov.rsplit('_', 1)[1]}_cmle"
            for label in W.SEQUENCING:
                (folder / f"{stem}_{label}.tif").write_bytes(b"")
            for extra in EXTRA:
                (folder / extra.format(stem=stem)).write_text("not an image\n")
    return root


def listing(root):
    """Every path under root with its size and modification time: the input folder as the plan found it."""
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.rglob("*"))}


def raw_config(tmp_path, fovs=("tile_1", "tile_2"), **keys):
    """A configuration on the raw layout (processing, a regular grid's stitch) whose FOV list lies in
    ``<tmp_path>/lists/fovs.csv``; the list is written, the key is not set."""
    config = W.dataset_keys(tmp_path, processing=copy.deepcopy(X.PROCESSING), stitching=copy.deepcopy(GRID), **keys)
    raw_inputs(config, fovs)
    lists = tmp_path / "lists"
    lists.mkdir()
    (lists / "fovs.csv").write_text(fov_rows(fovs))
    return config, lists / "fovs.csv"


# --- the key -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("form", ["absolute", "relative"])
def test_fov_list_names_a_list_outside_the_input_folder(tmp_path, form):
    """The key's path form: fov_list absolute, or relative to the input root, names the FOV list; the plan reads the
    membership from it, lists it as the file fovs.csv by its absolute path (outside the input root), the stitch task
    declares that path, and the input folder, which holds no fovs.csv, is left as it was."""
    config, path = raw_config(tmp_path)
    root = W.input_root(config)
    config["fov_list"] = str(path) if form == "absolute" else "../../../lists/fovs.csv"
    assert _fov_list(config).resolve() == path.resolve()
    before = listing(root)
    _, manifest = W.plan(tmp_path, f"key-{form}", config)
    assert manifest["membership"]["fovs"] == ["tile_1", "tile_2"]
    assert manifest["membership"]["samples"] == {"tissue-2D": ["tile_1", "tile_2"]}
    listed = manifest["inputs"]["files"]["fovs.csv"]
    assert Path(listed["path"]).resolve() == path.resolve() and listed["bytes"] == path.stat().st_size
    (stitch,) = [t for t in manifest["tasks"] if t["kind"] == "stitch"]
    assert listed["path"] in stitch["inputs"] and "input:fovs.csv" not in stitch["inputs"]
    assert not (root / "fovs.csv").exists()
    assert listing(root) == before
    resolved = json.loads((W.output_root(config) / "workflow" / "resolved.json").read_text())
    assert resolved["configuration"]["fov_list"] == config["fov_list"]


@pytest.mark.parametrize("given", ["absent", "null"])
def test_without_fov_list_the_input_folder_s_fovs_csv_is_read(tmp_path, given):
    """The absent key: without fov_list (or with null) the plan reads <input_root>/fovs.csv as before and lists it
    relative to the input root; a list elsewhere is not read, and the resolved configuration writes no fov_list."""
    config, path = raw_config(tmp_path)
    path.write_text(fov_rows(["tile_2"]))                  # another list, which must not be read
    (W.input_root(config) / "fovs.csv").write_text(fov_rows(["tile_1", "tile_2"]))
    if given == "null":
        config["fov_list"] = None
    _, manifest = W.plan(tmp_path, f"default-{given}", config)
    assert manifest["membership"]["fovs"] == ["tile_1", "tile_2"]
    assert manifest["inputs"]["files"]["fovs.csv"]["path"] == "fovs.csv"
    (stitch,) = [t for t in manifest["tasks"] if t["kind"] == "stitch"]
    assert "input:fovs.csv" in stitch["inputs"]
    resolved = json.loads((W.output_root(config) / "workflow" / "resolved.json").read_text())
    assert resolved["configuration"].get("fov_list") is None
    assert ("fov_list" in resolved["configuration"]) == (given == "null")


@pytest.mark.parametrize("value, named", [("", "fov_list"), (3, "fov_list"), ("missing/fovs.csv", "does not exist")])
def test_a_bad_or_missing_fov_list_is_refused(tmp_path, value, named):
    """An empty or non-string fov_list is refused by the schema, naming the key, and a list that does not exist by the
    membership check, naming the file; the plan writes nothing."""
    config, _ = raw_config(tmp_path, fov_list=value)
    with pytest.raises(PlanRefused) as refusal:
        W.plan(tmp_path, "bad", config)
    assert named in str(refusal.value)
    if value == "missing/fovs.csv":
        assert str(W.input_root(config) / "missing" / "fovs.csv") in str(refusal.value)
    assert not (W.output_root(config) / "workflow").exists()


# --- the raw folder read in place --------------------------------------------------------------------------------

def test_files_of_no_configured_channel_are_ignored(tmp_path):
    """Ignored files: in the raw layout each label matches its one channel file, whose name holds spaces; the .hgsb
    and _history.txt files beside it are neither listed nor declared by any task, and nothing is read from them."""
    config, path = raw_config(tmp_path)
    config["fov_list"] = str(path)
    _, manifest = W.plan(tmp_path, "ignored", config)
    images = manifest["inputs"]["images"]
    assert list(images) == ["tile_1", "tile_2"]
    for fov, rounds in images.items():
        n = fov.rsplit("_", 1)[1]
        assert {name: {label: entry["path"] for label, entry in labels.items()} for name, labels in rounds.items()} \
            == {f"round{r}": {label: f"round{r}/{fov}/9919seqA{r}_TileScan_001_Tile {n}_cmle_{label}.tif"
                              for label in W.SEQUENCING} for r in (1, 2, 3)}
    declared = [i for t in manifest["tasks"] for i in t["inputs"]]
    assert not [i for i in declared if i.endswith((".hgsb", ".txt"))]
    assert len({i for i in declared if i.startswith("input:") and i.endswith(".tif")}) == 2 * 3 * 4


def test_a_second_file_of_one_channel_is_refused_naming_both(tmp_path):
    """A duplicate channel file: a second file that matches a label in a FOV's round folder is refused before any
    task, naming the FOV, the round, the label and both files."""
    config, path = raw_config(tmp_path)
    config["fov_list"] = str(path)
    folder = W.input_root(config) / "round2" / "tile_2"
    first = "9919seqA2_TileScan_001_Tile 2_cmle_ch01.tif"
    (folder / "9919seqA2_TileScan_001_Tile 2_cmle_ch01 copy.tif").write_bytes(b"")
    with pytest.raises(PlanRefused) as refusal:
        W.plan(tmp_path, "duplicate", config)
    message = str(refusal.value)
    for part in ("FOV tile_2", "round round2", "'ch01'", "found 2", first,
                 "9919seqA2_TileScan_001_Tile 2_cmle_ch01 copy.tif"):
        assert part in message, (part, message)
    assert not (W.output_root(config) / "workflow").exists()


# --- the grid order ----------------------------------------------------------------------------------------------

def test_a_lexical_list_of_numbered_fovs_warns_naming_the_sample(tmp_path):
    """The order warning: rows tile_1, tile_10, tile_11, tile_12, tile_2, ... (lexical, where the numbers give
    another order) make the plan warn, naming the list, the sample and both orders; selected_fovs does not hide it
    (the grid places the whole list), and the plan still writes its files."""
    config, path = raw_config(tmp_path, fovs=NUMERIC, selected_fovs=["tile_1", "tile_2"])
    config["fov_list"] = str(path)
    config["stitching"]["positions"]["regular_grid"].update(columns=3, rows=4)
    path.write_text(fov_rows(LEXICAL))
    with pytest.warns(UserWarning) as caught:
        W.plan(tmp_path, "lexical", config)
    (warning,) = [w for w in caught if "lexical order" in str(w.message)]
    message = str(warning.message)
    assert str(path) in message and "sample 'tissue-2D'" in message
    assert "(tile_1, tile_10, tile_11, tile_12, ...)" in message and "(tile_1, tile_2, tile_3, tile_4, ...)" in message
    assert "regular_grid" in message
    assert (W.output_root(config) / "workflow" / "manifest.json").is_file()
    assert _lexical_order(config) == {"tissue-2D": (LEXICAL, NUMERIC)}


@pytest.mark.parametrize("rows", [
    NUMERIC,                                                        # scan order
    ["tile_2", "tile_1", *NUMERIC[2:]],                             # another order, not lexical
    [f"tile_{i:02d}" for i in range(1, 13)],                        # zero-padded: lexical is numeric
    ["A1", "A10", "A2", "B1"],                                      # no single numeric pattern
    ["tile_1"],                                                     # one FOV
], ids=["numeric", "custom", "padded", "no-pattern", "one"])
def test_no_warning_without_a_lexical_order_that_differs(tmp_path, rows):
    """No order warning for a list in numeric order, in an order of its own, of zero-padded IDs (lexical and numeric
    agree), of IDs that follow no single numeric pattern, or of one FOV."""
    config = W.dataset_keys(tmp_path, processing=copy.deepcopy(X.PROCESSING))
    W.write_inputs(config, fov_rows(rows))
    assert _lexical_order(config) == {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        W.plan(tmp_path, "quiet", config)


def test_the_order_is_checked_per_sample(tmp_path):
    """Each sample's rows are checked on their own: a sample in lexical order is named, one in numeric order is not,
    even when their rows interleave."""
    rows = [("s1", "F1"), ("s2", "F1x"), ("s1", "F10"), ("s2", "F2x"), ("s1", "F2"), ("s2", "F10x")]
    config = W.dataset_keys(tmp_path, processing=copy.deepcopy(X.PROCESSING))
    W.write_inputs(config, "sample,fov_id\n" + "".join(f"{s},{f}\n" for s, f in rows))
    assert _membership(config).samples == {"s1": ("F1", "F10", "F2"), "s2": ("F1x", "F2x", "F10x")}
    assert _lexical_order(config) == {"s1": (["F1", "F10", "F2"], ["F1", "F2", "F10"])}
