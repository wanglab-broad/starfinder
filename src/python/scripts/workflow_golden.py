"""Record and compare the golden baseline of the current Snakemake workflow (W-642, design/baselines/orchestration.md).

The baseline that workflow orchestration (§2.13) reproduces (D10): the current workflow with ``backend: python`` run
by an installed Snakemake 9 on a synthetic fixture, and the explicit expected values of its outputs, written in the
convention of ``test/expected_values.py`` (docs/contributing.md, "Expected values and tolerances").

Subcommands, each run in its own process:

``fixture WORK``
    Generate the fixture with the package's generator (``sample_scene_preset("sample-tiny-v1")``, seed 42: 2 × 2 FOVs
    of 5 × 36 × 36, four rounds of four channels) and write the workflow inputs under ``WORK/input/``: one OME-TIFF
    per round, FOV and channel, the nuclear stain as the reference round's ``ch04.tif``, an additional round
    ``flamingo`` (``ch04.tif`` the stain, ``ch05.tif`` the reference round's channel maximum), these three as uint16, ``genes.csv``,
    ``fovs.csv`` and ``positions.csv``; the truth cell labels under ``WORK/labels/`` as plain ZYX TIFFs without
    stored metadata; ``WORK/fixture.json`` lists
    every file with its SHA-256. Run with the package's environment.
``run WORK --snakemake EXE``
    Write the two configurations and run the workflow in ``WORK/output/``: invocation A (``config-a.yaml``, target
    ``all``: ``rsf_single_fov``, ``rotate_nuclei``, ``nuclei_registration``, ``create_nuclei_amplicon_overlay``,
    ``report_sample`` and ``report_dataset``), invocation A2 (the same configuration, target the files of
    ``enhance_dapi_with_flamingo``, whose inputs ``nuclei_registration`` writes without declaring them, so that they
    must exist before the DAG is built), then the truth labels copied as the
    legacy label files ``images/stardist_segmentation/<fov>.tif``, then invocation B (``config-b.yaml``, which adds
    the stitching, assembly and export blocks; target the saved assignment of each FOV, ``reads_assignment``, with
    ``stardist_segmentation`` not allowed), then invocation C (``config-c.yaml``, a stitching block of method
    ``bigstitcher``; target the macro and view map of ``bigstitcher_macro_sample``, the one rule allowed: its engine
    rule runs Fiji). ``WORK/commands.json`` records the commands and their exit status.
``extract WORK OUT``
    Write the expected values of the outputs to ``OUT`` (``values.json``, tables with ``dtypes.json``, ``arrays.npz``).
    Reads the outputs with NumPy, pandas, tifffile and json only.
``run-export WORK --snakemake EXE`` (W-643)
    Invocation D, on a fixture of its own: stitching with ``multiview_stitcher`` (B's positions table, the binning
    ``(1, 2, 2)`` and the link filter of the package's stitching tests), the export on each route the fixture allows
    and the report after the export, each route in its own output root (``golden-per-fov``, ``golden-molecules-only``,
    ``golden-after-fusion``) with target ``all``. The per-FOV route exports the saved assignments of
    ``reads_assignment`` (the truth labels copied first, as in B) with the fused cell labels; their Z maximum, written
    on the fused projection grid, is the import file of the route after fusion's one cell run. Records the commands in
    ``WORK/commands-d.json``.
``extract-export WORK OUT`` (W-643)
    The expected values of invocation D in the layout of ``extract``: the records, tables and images of each route's
    ``stitching/``, ``export/`` and ``report/`` folders, the content of each ``sample.h5ad`` (X, layers, obsm, obs,
    var and the normalized uns) and the import file; its per-FOV outputs come from the configurations of A and B,
    which pin them. Run with an environment that has anndata.
``compare EXPECTED ACTUAL``
    Compare two ``extract`` (or two ``extract-export``) folders; print the differences and exit 1 when there is one.

The comparison classes of ``values.json`` (docs/contributing.md): ``exact`` (integers, counts, strings, statuses,
booleans, integer images, the fixture's file hashes), ``float64`` (rtol 1e-9, atol 1e-12), ``float32`` (rtol 1e-6,
atol 1e-6) and ``same_environment`` (the SHA-256 of an output file's bytes: identity in one environment, which a run
in another environment does not have to reproduce). Time, host, version and absolute-path fields are left out of the
records (``NORMALIZED``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "test"))
import expected_values as EV  # noqa: E402

PRESET = "sample-tiny-v1"
DATASET_ID = PRESET
SAMPLE_ID = "sample"
OUTPUT_ID = "golden"
ROUNDS = ("round1", "round2", "round3", "round4")
CHANNELS = ("ch00", "ch01", "ch02", "ch03")
STAIN = "ch04"
FLAMINGO = "ch05"
ASSIGNMENT = "default"
# The record fields that say where, when and by which software a run was made, and its measured times and memory
# (the step seconds, wall_s, cpu_s, peak_rss_kib): a key of this set is left out wherever it occurs in a JSON record.
# workflow_config holds the configuration file's absolute path and the SHA-256 of its bytes, which hold the run's
# absolute paths.
NORMALIZED = frozenset({
    "started_at", "ended_at", "generated_at", "code", "environment", "software", "input_root", "output_root",
    "workflow_config", "seconds", "wall_s", "cpu_s", "peak_rss_kib"})


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def input_root(work):
    return Path(work) / "input" / DATASET_ID / SAMPLE_ID


def output_root(work):
    return Path(work) / "output" / DATASET_ID / OUTPUT_ID


# --- fixture ------------------------------------------------------------------------------------------------------

def _uint16(image):
    """A float image rounded and clipped to uint16."""
    return np.ascontiguousarray(np.clip(np.rint(image), 0, np.iinfo(np.uint16).max).astype(np.uint16))


def write_fixture(work):
    """The scene, the workflow inputs and the truth labels under ``work`` (a new or empty folder)."""
    import tifffile
    from starfinder.io import save_volume
    from starfinder.readout import load_codebook
    from starfinder.readout._encoding import colors_to_bases
    from starfinder.synthetic import generate_sample_scene, sample_scene_preset, save_sample_scene
    work = Path(work)
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"{work} is not empty")
    book, config = sample_scene_preset(PRESET)
    scene = generate_sample_scene(book, config=config)
    save_sample_scene(scene, work / "scene")
    if tuple(book.round_labels) != ROUNDS or tuple(book.channel_labels) != CHANNELS:
        raise ValueError(f"unexpected preset labels {book.round_labels} {book.channel_labels}")
    root = input_root(work)
    fov_ids = list(scene.positions.fov_id)
    for fov in fov_ids:
        formed, metadata = scene.fovs[fov], scene.fov_metadata[fov]
        for r in ROUNDS:
            for c, channel in enumerate(CHANNELS):
                save_volume(formed.rounds[r][..., c:c + 1], root / r / fov / f"{channel}.ome.tif", metadata=metadata)
        stain = _uint16(scene.stains[fov])
        # Plain uint16 ZYX TIFFs, as a microscope writes them: rotate_nuclei globs <ref_round>/<fov>/*ch04.tif and
        # reads it with skimage, and enhance_with_flamingo takes integer images only.
        for folder, name, image in ((ROUNDS[0], STAIN, stain), ("flamingo", STAIN, stain),
                                    ("flamingo", FLAMINGO, _uint16(formed.rounds[ROUNDS[0]].max(axis=-1)))):
            (root / folder / fov).mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(root / folder / fov / f"{name}.tif", image, photometric="minisblack")
        # The truth cells as a legacy label file: a plain ZYX TIFF without stored metadata, as stardist_segmentation
        # writes it, which reads_assignment imports on the FOV's reference grid.
        (work / "labels").mkdir(parents=True, exist_ok=True)
        cells = tifffile.imread(work / "scene" / "fovs" / fov / "labels" / "cells.tif")
        tifffile.imwrite(work / "labels" / f"{fov}.tif", np.ascontiguousarray(cells.reshape(config.fov.shape_zyx)),
                         compression="zlib", photometric="minisblack")
    # genes.csv: headerless gene,barcode rows of the two_base encoding the workflow loads by default.
    book_rows = book.table.drop_duplicates("gene_id")
    lines = [f"{gene},{colors_to_bases(colors, 'C')[::-1]}"
             for gene, colors in zip(book_rows.gene_id, book_rows.color_sequence)]
    (root / "genes.csv").write_text("\n".join(lines) + "\n")
    loaded = load_codebook(root / "genes.csv", round_labels=ROUNDS, channel_labels=CHANNELS)
    if dict(loaded.gene_to_seq) != dict(book.gene_to_seq):
        raise ValueError("genes.csv does not load as the preset's codebook")
    (root / "fovs.csv").write_text("sample,fov_id\n" + "".join(f"{SAMPLE_ID},{fov}\n" for fov in fov_ids))
    shutil.copyfile(work / "scene" / "positions.csv", root / "positions.csv")
    files = {p.relative_to(work).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in sorted(work.rglob("*")) if p.is_file()}
    record = {"preset": PRESET, "seed": config.fov.seed, "fov_ids": fov_ids, "shape_zyx": list(config.fov.shape_zyx),
              "rounds": list(ROUNDS), "channels": list(CHANNELS), "files": files}
    (work / "fixture.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


# --- configurations and the run -----------------------------------------------------------------------------------

def configurations(work, starfinder_path):
    """(config A, config B, config C) of the run: B is A with the stitching, assembly and export blocks, without the
    report block; C is A with a stitching block of method bigstitcher, without the report block."""
    work = Path(work)
    rules = {
        "rsf_single_fov": {"run": True, "parameters": {
            "load_codebook": {"run": True},
            "preprocessing": {"steps": [
                {"method": "min_max_normalization", "output_dtype": "uint8", "output_range": [0, 255],
                 "rounding": "truncate"},
                {"method": "white_tophat", "radius_yx": 3}]},
            "registration": {"steps": [{"method": "translation"}], "signal": {"mode": "max"}},
            "spot_finding": {"run": True, "method": "local_maxima", "exclude_border": False},
            "reads_extraction": {"run": True, "voxel_size": [1, 2, 2]},
            "reads_filtration": {"run": True, "end_base": "CC", "start_base": "C"},
            "decoding": {"method": "wta", "negative_policy": "clip_negative"},
            "deduplication": {"run": True},
            "checkpoints": {"stages": ["candidates", "pre_qc"]}}},
        "rotate_nuclei": {"run": True},
        "nuclei_registration": {"run": True},
        "create_nuclei_amplicon_overlay": {"run": True, "parameters": {"maximum_projection": False}},
        # Not in the target list of all: its inputs, the channel images nuclei_registration writes, are no declared
        # output of any rule, so the DAG of all would miss them (invocation A2 asks for its files).
        "enhance_dapi_with_flamingo": {"run": False},
        "reads_assignment": {"run": False, "parameters": {"expand_labels": False}},
    }
    base = {
        "starfinder_path": str(starfinder_path), "root_input_path": str(work / "input"),
        "root_output_path": str(work / "output"), "dataset_id": DATASET_ID, "sample_id": SAMPLE_ID,
        "output_id": OUTPUT_ID, "backend": "python", "workflow_mode": "free", "n_rounds": len(ROUNDS),
        "ref_round": ROUNDS[0], "ref_channel": "DAPI", "rotate_angle": 0, "maximum_projection": False,
        "voxel_size_xy": 0.5, "voxel_size_z": 1.0, "seq_channel_order": list(CHANNELS),
        "reference_stains": [{"channel": STAIN, "name": "DAPI"}],
        "additional_round": [{"round_name": "flamingo", "channel_order": [
            {"channel": STAIN, "name": "DAPI"}, {"channel": FLAMINGO, "name": "Flamingo"}]}],
        "rules": rules}
    a = dict(base, config_path=str(work / "config-a.yaml"), report={})
    rules_b = json.loads(json.dumps(rules))
    rules_b["reads_assignment"]["run"] = True
    b = dict(base, config_path=str(work / "config-b.yaml"), rules=rules_b,
             stitching={"method": "multiview_stitcher",
                        "positions": {"source": "table", "table": str(input_root(work) / "positions.csv")}},
             assembly={"assignment": ASSIGNMENT},
             # The default fusion dimensionality (projection) resolves to the route after fusion, which segments
             # the fused image; the route per FOV assembles the saved assignments that reads_assignment writes.
             export={"route": "per_fov"})
    # C: A with the stitching block of method bigstitcher on a regular grid (the macro rule needs grid cells), without
    # the report block; img_row and img_col, which the legacy tile configuration writer of the import requires.
    shape = json.loads((work / "fixture.json").read_text())["shape_zyx"]
    c = dict({k: v for k, v in base.items()}, config_path=str(work / "config-c.yaml"), img_row=shape[1],
             img_col=shape[2],
             stitching={"method": "bigstitcher", "bigstitcher_save_dir": str(work / "bigstitcher"),
                        "positions": {"source": "regular_grid", "regular_grid": {
                            "columns": 2, "rows": 2, "order": "column_major_down_right", "overlap": 1 / 3}}})
    return a, b, c


def run_workflow(work, snakemake, starfinder_path, cores=1):
    """Invocations A and B (see the module docstring); returns the command records."""
    import yaml
    work = Path(work)
    fov_ids = json.loads((work / "fixture.json").read_text())["fov_ids"]
    a, b, c = configurations(work, starfinder_path)
    for config in (a, b, c):
        Path(config["config_path"]).write_text(yaml.safe_dump(config, sort_keys=False))
    snakefile = Path(starfinder_path) / "workflow" / "Snakefile"
    common = [str(snakemake), "-s", str(snakefile), "--directory", str(work / "snakemake"), "--cores", str(cores),
              "--nocolor", "--printshellcmds"]
    environment = dict(os.environ, XDG_CACHE_HOME=str(work / "cache"), PYTHONDONTWRITEBYTECODE="1")
    out = output_root(work)
    allowed = ["rsf_single_fov", "rotate_nuclei", "create_nuclei_amplicon_overlay", "nuclei_registration",
               "enhance_dapi_with_flamingo", "reads_assignment"]
    targets = [str(out / "fovs" / fov / "assignment" / ASSIGNMENT / "assignment.json") for fov in fov_ids]
    records = []

    def invoke(name, command):
        log = work / f"snakemake-{name}.log"
        with open(log, "w") as handle:
            status = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=environment).returncode
        records.append({"name": name, "command": command, "exit_status": status, "log": log.name})
        (work / "commands.json").write_text(json.dumps(records, indent=1) + "\n")
        if status:
            raise SystemExit(f"invocation {name} exited with status {status}; see {log}")

    # --configfile and --allowed-rules take several values, so "--" ends them before the targets.
    invoke("A", common + ["--configfile", a["config_path"], "--", "all"])
    invoke("A2", common + ["--configfile", a["config_path"], "--",
                           *[str(out / "images" / "flamingo" / "enhanced_DAPI" / f"{fov}.tif") for fov in fov_ids]])
    for fov in fov_ids:
        labels = out / "images" / "stardist_segmentation" / f"{fov}.tif"
        labels.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(work / "labels" / f"{fov}.tif", labels)
    invoke("B", common + ["--configfile", b["config_path"], "--allowed-rules", *allowed, "--", *targets])
    # The BigStitcher macro rule alone (the tiles, the macro and the view map); its engine rule runs Fiji.
    folder = Path(c["stitching"]["bigstitcher_save_dir"]) / SAMPLE_ID
    invoke("C", common + ["--configfile", c["config_path"], "--allowed-rules", "bigstitcher_macro_sample", "--",
                          str(folder / "BigStitcher_macro.ijm"), str(folder / "view_map.csv")])
    return records


# --- invocation D: stitching, the export on each route and the report after the export (W-643) ----------------------

# The routes of export_sample the fixture runs without a learned method: per FOV (the saved assignments of
# reads_assignment), molecules only (assembly.assignment null) and after fusion (the fused projection segmented by an
# import run of the per-FOV route's fused cell labels). Each route runs in its own output root, golden-<route>.
EXPORT_ROUTES = ("per_fov", "molecules_only", "after_fusion")
# The per-FOV route's fused cell labels (export/<sample>/cell_labels.ome.tif, fuse_labels at the fusion's scale), copied
# as the label file that the route after fusion imports on the fused projection grid.
FUSED_CELLS = "fused-cells.ome.tif"
# The rules invocation D may run: those of A and B, the stitching, export and report rules.
EXPORT_RULES = ("rsf_single_fov", "rotate_nuclei", "create_nuclei_amplicon_overlay", "nuclei_registration",
                "reads_assignment", "stitch_sample", "export_sample", "report_sample", "report_dataset")


def export_output_root(work, route):
    return Path(work) / "output" / DATASET_ID / f"{OUTPUT_ID}-{route.replace('_', '-')}"


def export_configurations(work, starfinder_path):
    """{route: configuration} of invocation D: each is configuration A (its processing and morphology rules) with B's
    stitching block of method multiview_stitcher, an assembly block, an export block and the report block, in its
    own output root. The export fuses the reference image (``images: [reference_merged]``, a declared output of
    rsf_single_fov; the default ``reference_stain`` reads the undeclared saved form of the morphology rounds)."""
    work = Path(work)
    a, b, _ = configurations(work, starfinder_path)
    out = {}
    for route in EXPORT_ROUTES:
        name = route.replace("_", "-")
        # The stitching of B with the binning and link filter of the package's stitching tests: the default binning
        # (4 in Z) leaves one plane of the 5-plane FOVs, on which multiview_stitcher's overlap is undefined.
        stitching = dict(b["stitching"], config={"binning_zyx": [1, 2, 2], "quality_threshold": -1.0})
        config = dict(a, config_path=str(work / f"config-d-{name}.yaml"), output_id=f"{OUTPUT_ID}-{name}",
                      stitching=stitching, report={})
        if route == "per_fov":
            # B's rules (reads_assignment saves each FOV's assignment); the fused cell labels at the fusion's scale.
            config.update(rules=b["rules"], assembly={"assignment": ASSIGNMENT},
                          export={"route": "per_fov", "images": ["reference_merged"], "fuse_labels": True,
                                  "labels_scale_yx": 0.5})
        elif route == "molecules_only":
            config.update(assembly={"assignment": None}, export={"images": ["reference_merged"]})
        else:
            # The default route of dimensionality projection; one cell run imported on the fused projection grid.
            config.update(assembly={"assignment": ASSIGNMENT}, export={
                "images": ["reference_merged"],
                "segmentation": {"cells": "cell", "runs": [
                    {"name": "cell", "target": "cell", "method": "import", "path": str(work / "labels" / FUSED_CELLS)}]}})
        out[route] = config
    return out


def _fused_cells(export_dir, target):
    """The per-FOV route's fused cell labels as the import file of the route after fusion: their Z maximum, written as a
    uint32 OME-TIFF with the metadata of the fused projection of the same export (the segmentation grid of the route
    after fusion is that projection's; the label export's own grid is a stack whose origin differs by a fraction of a
    pixel)."""
    import numpy as np
    from starfinder.io import _ome, load_volume
    labels = load_volume(Path(export_dir) / "cell_labels.ome.tif").image
    grid = load_volume(Path(export_dir) / "reference_merged_projection.ome.tif")
    if labels.shape[-2:] != grid.image.shape[-2:]:
        raise ValueError(f"the fused cell labels {labels.shape} and the fused projection {grid.image.shape} differ in YX")
    _ome.write_ome_tiff(target, np.ascontiguousarray(labels.max(axis=0)).astype(np.uint32), metadata=grid.metadata,
                        annotation={"golden": "the Z maximum of the per-FOV route's fused cell labels"})


def run_export(work, snakemake, starfinder_path, cores=1):
    """Invocation D (the module docstring): D-per-fov, then the fused cell labels copied, D-molecules-only and
    D-after-fusion, each with target all; returns the command records (``WORK/commands-d.json``)."""
    import yaml
    work = Path(work)
    fov_ids = json.loads((work / "fixture.json").read_text())["fov_ids"]
    configs = export_configurations(work, starfinder_path)
    for config in configs.values():
        Path(config["config_path"]).write_text(yaml.safe_dump(config, sort_keys=False))
    snakefile = Path(starfinder_path) / "workflow" / "Snakefile"
    common = [str(snakemake), "-s", str(snakefile), "--directory", str(work / "snakemake"), "--cores", str(cores),
              "--nocolor", "--printshellcmds", "--allowed-rules", *EXPORT_RULES]
    environment = dict(os.environ, XDG_CACHE_HOME=str(work / "cache"), PYTHONDONTWRITEBYTECODE="1")
    records = []

    def invoke(name, config):
        log = work / f"snakemake-{name}.log"
        command = common + ["--configfile", config["config_path"], "--", "all"]
        with open(log, "w") as handle:
            status = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=environment).returncode
        records.append({"name": name, "command": command, "exit_status": status, "log": log.name})
        (work / "commands-d.json").write_text(json.dumps(records, indent=1) + "\n")
        if status:
            raise SystemExit(f"invocation {name} exited with status {status}; see {log}")

    root = export_output_root(work, "per_fov")
    for fov in fov_ids:
        labels = root / "images" / "stardist_segmentation" / f"{fov}.tif"
        labels.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(work / "labels" / f"{fov}.tif", labels)
    invoke("D-per-fov", configs["per_fov"])
    _fused_cells(root / "export" / SAMPLE_ID, work / "labels" / FUSED_CELLS)
    invoke("D-molecules-only", configs["molecules_only"])
    invoke("D-after-fusion", configs["after_fusion"])
    return records


# --- the expected values ------------------------------------------------------------------------------------------

# Output files whose bytes hold a time, a measured duration or an absolute path of the run (the run records, the
# report's pages and its time columns, a saved assignment's record, the Snakemake benchmark logs, and the channel
# images of nuclei_registration, whose stored frame_id is the absolute path of the reference stain file, and the
# BigStitcher macro, which names the tiles' absolute folder): their
# SHA-256 and byte count are left out wherever a record lists them, and they have no file digest.
VARYING = ("run.json", "report.json", "report.html", "index.html", "assignment.json", "samples.csv",
           "tables/fovs.csv", "tables/steps.csv", "log/benchmark/", "images/flamingo/ch", "BigStitcher_macro.ijm")
# The table columns that hold times, durations and the digests of varying files (the report's tables).
VARYING_COLUMNS = frozenset({"started_at", "ended_at", "run_seconds", "seconds", "generated_at", "report_sha256"})
# The JSON keys of digests of records that hold an absolute path (an assignment's input records).
VARYING_DIGESTS = frozenset({"record_sha256"})


def _varies(path, varying=VARYING):
    return any(pattern in str(path) for pattern in varying)


def _normalized(value, work, varying=VARYING, digests=VARYING_DIGESTS):
    """A JSON record without the NORMALIZED keys, the digests of varying records and files, with the work folder's
    absolute path written as ``<work>`` wherever a string holds it."""
    if isinstance(value, dict):
        names = value.get("path")
        drop = {"sha256", "bytes"} if isinstance(names, str) and _varies(names, varying) else set()
        return {key: _normalized(item, work, varying, digests) for key, item in value.items()
                if key not in NORMALIZED and key not in digests and key not in drop}
    if isinstance(value, list):
        return [_normalized(item, work, varying, digests) for item in value]
    if isinstance(value, str):
        return value.replace(work, "<work>")
    return value


def _table(path):
    """A CSV output as an expected table: string, int64, float64 or bool columns as pandas infers them, without the
    VARYING_COLUMNS."""
    frame = pd.read_csv(path, keep_default_na=False, na_values=["", "<NA>"])
    out = {}
    for column in frame.columns:
        if column in VARYING_COLUMNS:
            continue
        series = frame[column]
        if series.dtype.kind in "iu":
            out[column] = series.astype("int64")
        elif series.dtype.kind == "f":
            out[column] = series.astype("float64")
        elif series.dtype.kind == "b":
            out[column] = series.astype(bool)
        else:
            out[column] = series.astype("string")
    return pd.DataFrame(out)


def _image(path):
    import tifffile
    with tifffile.TiffFile(path) as tif:
        return tif.series[0].asarray()


def extract(work, out):
    """The expected values of one run in ``out``: ``records/`` (each JSON record normalized), the tables with their
    ``dtypes.json`` (the report's intensity histograms summarized, one row per histogram), ``arrays.npz`` (each FOV's
    reference image) and ``values.json`` (the fixture's file digests; the shape, dtype, minimum, maximum and sum of
    the other images and of the BigStitcher tiles; the macro's text; and the output files' digests)."""
    work, out = Path(work).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fixture = json.loads((work / "fixture.json").read_text())
    root = output_root(work)
    values, dtypes, arrays = {}, {}, {}
    for name, entry in fixture["files"].items():
        values[f"fixture:{name}"] = {"value": entry["sha256"], "compare": "exact"}
    records = [p for fov in fixture["fov_ids"] for p in (root / "fovs" / fov / "run.json",
                                                         root / "fovs" / fov / "nuclei_registration.json",
                                                         root / "fovs" / fov / "assignment" / ASSIGNMENT /
                                                         "assignment.json")]
    records += sorted((root / "report").rglob("*.json"))
    for path in records:
        target = out / "records" / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(_normalized(json.loads(path.read_text()), str(work)), indent=1) + "\n")
    tables = {}
    for fov in fixture["fov_ids"]:
        folder = root / "fovs" / fov
        tables[f"{fov}-molecules.csv"] = folder / "molecules.csv"
        for path in sorted((folder / "assignment" / ASSIGNMENT).glob("*.csv")):
            tables[f"{fov}-assignment-{path.name}"] = path
        reference = _image(folder / "reference.tif")
        if reference.dtype.kind not in "iu":
            raise ValueError(f"{folder / 'reference.tif'} is not an integer image")
        arrays[f"{fov}/reference"] = reference
        for folder_name in ("DAPI", "overlay", f"flamingo/{STAIN}", f"flamingo/{FLAMINGO}", "flamingo/enhanced_DAPI"):
            image = root / "images" / folder_name / f"{fov}.tif"
            name = image.relative_to(root).as_posix()
            pixels = _image(image)
            integer = pixels.dtype.kind in "iub"
            kind = "exact" if integer else "float32" if pixels.dtype == np.float32 else "float64"
            values[f"{name}:shape"] = {"value": list(pixels.shape), "compare": "exact"}
            values[f"{name}:dtype"] = {"value": str(pixels.dtype), "compare": "exact"}
            values[f"{name}:min"] = {"value": pixels.min().item(), "compare": kind}
            values[f"{name}:max"] = {"value": pixels.max().item(), "compare": kind}
            # A sum of integer pixels holds an integer value (exact); a sum of float pixels is a float64 value.
            values[f"{name}:sum"] = {"value": float(pixels.sum(dtype=np.float64)),
                                     "compare": "exact" if integer else "float64"}
    for path in sorted((root / "report").rglob("*.csv")):
        tables["report-" + path.relative_to(root / "report").as_posix().replace("/", "-")] = path
    bigstitcher = work / "bigstitcher" / SAMPLE_ID
    tables["bigstitcher-view_map.csv"] = bigstitcher / "view_map.csv"
    macro = (bigstitcher / "BigStitcher_macro.ijm").read_text()
    values["bigstitcher/BigStitcher_macro.ijm:text"] = {"value": macro.replace(str(work), "<work>"),
                                                         "compare": "exact"}
    for image in sorted((bigstitcher / "tiles").glob("*.tif")):
        name = image.relative_to(work).as_posix()
        pixels = _image(image)
        integer = pixels.dtype.kind in "iub"
        kind = "exact" if integer else "float32" if pixels.dtype == np.float32 else "float64"
        values[f"{name}:shape"] = {"value": list(pixels.shape), "compare": "exact"}
        values[f"{name}:dtype"] = {"value": str(pixels.dtype), "compare": "exact"}
        values[f"{name}:min"] = {"value": pixels.min().item(), "compare": kind}
        values[f"{name}:max"] = {"value": pixels.max().item(), "compare": kind}
        values[f"{name}:sum"] = {"value": float(pixels.sum(dtype=np.float64)), "compare": "exact" if integer else "float64"}
        values[f"file:{name}"] = {"value": sha256_file(image), "compare": "same_environment"}
    for name, path in tables.items():
        frame = _table(path)
        if name == "report-sample-tables-intensity.csv":
            # The intensity histograms (thousands of bins) as one row per histogram: its bins, its total count and
            # its range.
            name = "report-sample-tables-intensity-summary.csv"
            keys = ["round", "step_index", "phase", "step", "channel"]
            frame = frame.groupby(keys, sort=False).agg(bins=("bin", "size"), count=("count", "sum"),
                                                         bin_low=("bin_low", "min"),
                                                         bin_high=("bin_high", "max")).reset_index()
            frame["bins"] = frame["bins"].astype("int64")
        dtypes[name] = EV.write_csv(frame, out / name)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if not _varies(path.relative_to(root).as_posix()):
            values[f"file:{path.relative_to(root).as_posix()}"] = {"value": sha256_file(path),
                                                                    "compare": "same_environment"}
    (out / "values.json").write_text(json.dumps(values, indent=1, sort_keys=True) + "\n")
    # Each table's columns keep their order (expected_values.read_csv reads them in it).
    (out / "dtypes.json").write_text(json.dumps({name: dtypes[name] for name in sorted(dtypes)}, indent=1) + "\n")
    np.savez_compressed(out / "arrays.npz", **arrays)
    return values


# The sample folders invocation D adds to what A and B pin: the placement, the export and the report after the export.
# Its per-FOV outputs (run records, molecule tables, reference images, saved assignments, morphology images) come from
# the configurations of A and B and are pinned by their expected values, so they are not extracted again.
EXPORT_FOLDERS = ("stitching", "export", "report")
# Besides VARYING: the files of invocation D whose bytes hold an absolute path or a time (found by two runs in
# different folders): the placement record (the absolute paths of the positions table and of each FOV's reference
# image), the export record and the AnnData file, whose uns holds the export's records.
VARYING_EXPORT = VARYING + ("placement.json", "export.json", "sample.h5ad")
# Besides VARYING_DIGESTS: the digest of the export record's assembly section, which holds the AnnData file's digest.
VARYING_EXPORT_DIGESTS = VARYING_DIGESTS | {"assembly_sha256"}


def _image_values(values, arrays, name, path):
    """The shape, dtype, minimum, maximum and sum of an image; an integer image also whole in ``arrays``."""
    pixels = _image(path)
    integer = pixels.dtype.kind in "iub"
    kind = "exact" if integer else "float32" if pixels.dtype == np.float32 else "float64"
    values[f"{name}:shape"] = {"value": list(pixels.shape), "compare": "exact"}
    values[f"{name}:dtype"] = {"value": str(pixels.dtype), "compare": "exact"}
    values[f"{name}:min"] = {"value": pixels.min().item(), "compare": kind}
    values[f"{name}:max"] = {"value": pixels.max().item(), "compare": kind}
    values[f"{name}:sum"] = {"value": float(pixels.sum(dtype=np.float64)), "compare": "exact" if integer else "float64"}
    if integer:
        arrays[name] = pixels


def _frame_table(frame):
    """A DataFrame of an AnnData (obs or var) as an expected table: its index as the first column ``_index``, then
    its columns as ``_table`` types them (a categorical as strings, an integer column with a missing value as float64,
    a boolean column with a missing value as strings)."""
    out = {"_index": pd.array([str(v) for v in frame.index], dtype="string")}
    for column in frame.columns:
        series = frame[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype(object)
        missing = bool(series.isna().any())
        kind = series.dtype.kind
        if kind in "iu":
            out[str(column)] = series.astype("float64" if missing else "int64")
        elif kind == "f":
            out[str(column)] = series.astype("float64")
        elif kind == "b" and not missing:
            out[str(column)] = series.astype(bool)
        else:
            out[str(column)] = pd.array([None if pd.isna(v) else str(v) for v in series], dtype="string")
    return pd.DataFrame(out)


def _matrix_values(values, arrays, name, matrix):
    """A matrix of an AnnData (X, a layer, an obsm, varm or obsp entry): shape, dtype and format in ``values``, the
    matrix whole in ``arrays`` (a sparse one as its ``data``, ``indices`` and ``indptr``)."""
    import scipy.sparse
    if scipy.sparse.issparse(matrix):
        matrix = scipy.sparse.csr_matrix(matrix)
        values[f"{name}:format"] = {"value": "csr", "compare": "exact"}
        for part in ("data", "indices", "indptr"):
            arrays[f"{name}/{part}"] = np.asarray(getattr(matrix, part))
    elif isinstance(matrix, pd.DataFrame):
        values[f"{name}:format"] = {"value": "dataframe", "compare": "exact"}
        return _frame_table(matrix)
    else:
        matrix = np.asarray(matrix)
        values[f"{name}:format"] = {"value": "dense", "compare": "exact"}
        arrays[name] = matrix
    values[f"{name}:shape"] = {"value": [int(n) for n in matrix.shape], "compare": "exact"}
    values[f"{name}:dtype"] = {"value": str(matrix.dtype), "compare": "exact"}
    return None


def _uns_value(value):
    """An ``uns`` entry as JSON values: a string that holds a JSON object or list (the export's records) parsed, arrays
    as lists, nested mappings recursively."""
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "{[":
            try:
                return json.loads(text)
            except ValueError:
                return value
        return value
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): _uns_value(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [_uns_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_uns_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _anndata_values(path, name, values, arrays, frames, records_dir, work):
    """The content of a ``sample.h5ad`` (W-643, repair 1), read with anndata: ``X``, every layer and every obsm, varm
    and obsp entry whole (``_matrix_values``); ``obs`` and ``var`` as tables (``_frame_table``), with the obs and var
    names; and ``uns`` as a JSON record, its JSON strings parsed and normalized as the export's records are
    (``_normalized`` with VARYING_EXPORT and VARYING_EXPORT_DIGESTS: the fields of where, when and with which software
    a run was made and its times, the digests of the files that hold absolute paths and of the records over them, and
    the work folder written ``<work>``), written to ``records_dir``."""
    import anndata
    data = anndata.read_h5ad(path)
    _matrix_values(values, arrays, f"{name}/X", data.X)
    for group in ("layers", "obsm", "varm", "obsp", "varp"):
        mapping = getattr(data, group)
        values[f"{name}/{group}:keys"] = {"value": sorted(mapping.keys()), "compare": "exact"}
        for key in sorted(mapping.keys()):
            frame = _matrix_values(values, arrays, f"{name}/{group}/{key}", mapping[key])
            if frame is not None:
                frames[f"{name}-{group}-{key}".replace("/", "-")] = frame
    frames[f"{name}-obs".replace("/", "-")] = _frame_table(data.obs)
    frames[f"{name}-var".replace("/", "-")] = _frame_table(data.var)
    uns = _normalized(_uns_value(dict(data.uns)), work, VARYING_EXPORT, VARYING_EXPORT_DIGESTS)
    records_dir.mkdir(parents=True, exist_ok=True)
    (records_dir / f"{Path(path).name}.uns.json").write_text(json.dumps(uns, indent=1) + "\n")


def extract_export(work, out):
    """The expected values of invocation D in ``out``, in the layout of ``extract``: for each route, the JSON records
    of stitching/, export/ and report/ normalized (``records/<route>/...``), their CSV tables (the report's intensity
    histograms summarized, one row per histogram), every TIFF's shape, dtype, minimum, maximum and sum, each integer
    image whole in ``arrays.npz``, and the digest of every file that holds no time or absolute path; the fixture's
    file digests; and the import file of the route after fusion. The file digest of ``sample.h5ad`` is left out (its
    ``uns`` holds absolute paths and times), and its content is extracted instead (``_anndata_values``, repair 1):
    ``X``, the layers and the obsm entries whole, ``obs`` and ``var`` as tables, ``uns`` as a normalized record
    (``records/<route>/export/sample/sample.h5ad.uns.json``), which leaves out only what ``_normalized`` leaves out
    of the export's JSON records. Reads ``sample.h5ad`` with anndata (the golden scratch environment's)."""
    work, out = Path(work).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fixture = json.loads((work / "fixture.json").read_text())
    values, dtypes, arrays, tables, frames = {}, {}, {}, {}, {}
    for name, entry in fixture["files"].items():
        values[f"fixture:{name}"] = {"value": entry["sha256"], "compare": "exact"}
    _image_values(values, arrays, f"labels/{FUSED_CELLS}", work / "labels" / FUSED_CELLS)
    for route in EXPORT_ROUTES:
        root = export_output_root(work, route)
        for path in sorted(p for folder in EXPORT_FOLDERS for p in (root / folder).rglob("*") if p.is_file()):
            relative = path.relative_to(root).as_posix()
            name = f"{route}/{relative}"
            if path.suffix == ".json":
                target = out / "records" / route / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                record = _normalized(json.loads(path.read_text()), str(work), VARYING_EXPORT, VARYING_EXPORT_DIGESTS)
                target.write_text(json.dumps(record, indent=1) + "\n")
            elif path.suffix == ".csv":
                tables[f"{route}-{relative.replace('/', '-')}"] = path
            elif path.name.endswith((".tif", ".tiff")):
                _image_values(values, arrays, name, path)
            elif path.suffix == ".h5ad":
                _anndata_values(path, name, values, arrays, frames,
                                (out / "records" / route / relative).parent, str(work))
            if not any(pattern in relative for pattern in VARYING_EXPORT):
                values[f"file:{name}"] = {"value": sha256_file(path), "compare": "same_environment"}
    for name, path in tables.items():
        frame = _table(path)
        if name.endswith("-report-sample-tables-intensity.csv"):
            name = name.replace("-intensity.csv", "-intensity-summary.csv")
            keys = ["round", "step_index", "phase", "step", "channel"]
            frame = frame.groupby(keys, sort=False).agg(bins=("bin", "size"), count=("count", "sum"),
                                                         bin_low=("bin_low", "min"),
                                                         bin_high=("bin_high", "max")).reset_index()
            frame["bins"] = frame["bins"].astype("int64")
        dtypes[name] = EV.write_csv(frame, out / name)
    for name, frame in frames.items():
        dtypes[f"{name}.csv"] = EV.write_csv(frame, out / f"{name}.csv")
    (out / "values.json").write_text(json.dumps(values, indent=1, sort_keys=True) + "\n")
    (out / "dtypes.json").write_text(json.dumps({name: dtypes[name] for name in sorted(dtypes)}, indent=1) + "\n")
    np.savez_compressed(out / "arrays.npz", **arrays)
    return values


# --- the comparison -----------------------------------------------------------------------------------------------

TOLERANCES = {"float64": EV.FLOAT64, "float32": {"rtol": 1e-6, "atol": 1e-6}}
# The digest a run record holds of each checkpoint stage record: candidates.json names the software it ran with
# (detection_diagnostics.software), so the digest is an identity of the recording environment, like the file digests,
# and a comparison across environments skips it (W-643: NumPy 2.3.5 against 2.2.6 changed only that leaf).
STAGE_RECORD_DIGEST = re.compile(r"^records/fovs/[^/]+/run\.json\.checkpoints\.[A-Za-z_]+\.sha256: ")


def _close(actual, expected, kind):
    """Whether a value matches: exactly, or within the tolerance of its kind for two non-Boolean numbers."""
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (actual, expected))
    if kind in ("exact", "same_environment") or not numbers:
        return type(actual) is type(expected) and actual == expected
    return bool(np.isclose(actual, expected, equal_nan=True, **TOLERANCES[kind]))


def record_differences(actual, expected, where=""):
    """How a normalized record differs: floats within the float64 tolerance, every other leaf exactly."""
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
    return [] if _close(actual, expected, kind) else [f"{where}: {actual!r} != {expected!r} ({kind})"]


def compare(expected_dir, actual_dir, *, same_environment=True):
    """The differences between two ``extract`` folders, as readable lines (empty when they match)."""
    expected_dir, actual_dir = Path(expected_dir), Path(actual_dir)
    lines = []
    expected = json.loads((expected_dir / "values.json").read_text())
    actual = json.loads((actual_dir / "values.json").read_text())
    for key in sorted(set(expected) | set(actual)):
        if key not in actual or key not in expected:
            lines.append(f"{key}: only in {'expected' if key in expected else 'actual'}")
            continue
        e, a = expected[key], actual[key]
        if e["compare"] != a["compare"]:
            lines.append(f"{key}: comparison {a['compare']} != {e['compare']}")
        elif (e["compare"] != "same_environment" or same_environment) and not _close(a["value"], e["value"],
                                                                                       e["compare"]):
            lines.append(f"{key}: {a['value']!r} != {e['value']!r} ({e['compare']})")
    names = lambda folder: {p.relative_to(folder / "records").as_posix() for p in (folder / "records").rglob("*.json")}
    for name in sorted(names(expected_dir) | names(actual_dir)):
        e, a = expected_dir / "records" / name, actual_dir / "records" / name
        if not (e.is_file() and a.is_file()):
            lines.append(f"records/{name}: only in {'expected' if e.is_file() else 'actual'}")
            continue
        lines += [f"records/{name}{line}" for line in record_differences(json.loads(a.read_text()),
                                                                          json.loads(e.read_text()))
                  if same_environment or not STAGE_RECORD_DIGEST.match(f"records/{name}{line}")]
    dtypes_e = json.loads((expected_dir / "dtypes.json").read_text())
    dtypes_a = json.loads((actual_dir / "dtypes.json").read_text())
    for name in sorted(set(dtypes_e) | set(dtypes_a)):
        if name not in dtypes_a or name not in dtypes_e:
            lines.append(f"{name}: only in {'expected' if name in dtypes_e else 'actual'}")
            continue
        e, a = EV.read_table(expected_dir / name), EV.read_table(actual_dir / name)
        close = [c for c, d in dtypes_e[name].items() if d == "float64" and not _integral(e[c])]
        lines += [f"{name}: {line}" for line in EV.table_differences(a, e, close)]
    e, a = EV.read_arrays(expected_dir / "arrays.npz"), EV.read_arrays(actual_dir / "arrays.npz")
    for name in sorted(set(e) | set(a)):
        if name not in a or name not in e:
            lines.append(f"arrays {name}: only in {'expected' if name in e else 'actual'}")
        elif e[name].dtype != a[name].dtype or e[name].shape != a[name].shape:
            lines.append(f"arrays {name}: differ")
        elif e[name].dtype.kind == "f":
            # A float array (an AnnData matrix, repair 1) within the tolerance of its dtype, NaN equal to NaN.
            tolerance = TOLERANCES["float32" if e[name].dtype == np.float32 else "float64"]
            if not np.allclose(a[name], e[name], equal_nan=True, **tolerance):
                lines.append(f"arrays {name}: differ")
        elif not np.array_equal(e[name], a[name]):
            lines.append(f"arrays {name}: differ")
    return lines


def _integral(series):
    """Whether a float column holds integer values only (compared exactly, docs/contributing.md)."""
    values = series.to_numpy(np.float64)
    finite = np.isfinite(values)
    return bool(np.array_equal(values[finite], np.round(values[finite])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fixture").add_argument("work", type=Path)
    run = commands.add_parser("run")
    run.add_argument("work", type=Path)
    run.add_argument("--snakemake", required=True)
    run.add_argument("--starfinder-path", type=Path, default=REPO)
    run.add_argument("--cores", type=int, default=1)
    extracting = commands.add_parser("extract")
    extracting.add_argument("work", type=Path)
    extracting.add_argument("out", type=Path)
    run_d = commands.add_parser("run-export")
    run_d.add_argument("work", type=Path)
    run_d.add_argument("--snakemake", required=True)
    run_d.add_argument("--starfinder-path", type=Path, default=REPO)
    run_d.add_argument("--cores", type=int, default=1)
    extracting_d = commands.add_parser("extract-export")
    extracting_d.add_argument("work", type=Path)
    extracting_d.add_argument("out", type=Path)
    comparing = commands.add_parser("compare")
    comparing.add_argument("expected", type=Path)
    comparing.add_argument("actual", type=Path)
    comparing.add_argument("--other-environment", action="store_true",
                           help="skip the same_environment values (the file digests)")
    args = parser.parse_args(argv)
    if args.command == "fixture":
        record = write_fixture(args.work)
        print(f"{len(record['files'])} fixture files in {args.work}")
    elif args.command == "run":
        for record in run_workflow(args.work, args.snakemake, args.starfinder_path, args.cores):
            print(record["name"], record["exit_status"])
    elif args.command == "extract":
        print(f"{len(extract(args.work, args.out))} values in {args.out}")
    elif args.command == "run-export":
        for record in run_export(args.work, args.snakemake, args.starfinder_path, args.cores):
            print(record["name"], record["exit_status"])
    elif args.command == "extract-export":
        print(f"{len(extract_export(args.work, args.out))} values in {args.out}")
    else:
        lines = compare(args.expected, args.actual, same_environment=not args.other_environment)
        print("\n".join(lines) if lines else "no difference")
        return 1 if lines else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
