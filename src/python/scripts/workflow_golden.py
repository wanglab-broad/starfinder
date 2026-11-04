"""Record and compare the golden baseline of the Snakemake workflow (W-642, design/baselines/orchestration.md).

The baseline that workflow orchestration (§2.13) reproduces (D10): the workflow with ``backend: python`` as it was before
the dispatcher (W-737), run by an installed Snakemake 9 on a synthetic fixture, and the explicit expected values of its
outputs, written in the convention of ``test/expected_values.py`` (docs/contributing.md, "Expected values and
tolerances"). W-642 recorded invocations A, A2, B and C, W-643 invocation D, each with a subcommand of this script that
drove that workflow (``run``, ``run-export``). W-737 replaced that workflow with the dispatcher over the plan's manifest
and removed both subcommands: ``4583228`` is the last revision on which the expected values can be re-recorded (its
``run``, ``extract``, ``run-export`` and ``extract-export``). The refactored workflow never re-records them; G3 compares
its values with them (``run-migrated``, ``compare-migrated``).

Subcommands, each run in its own process:

``fixture WORK``
    Generate the fixture with the package's generator (``sample_scene_preset("sample-tiny-v1")``, seed 42: 2 × 2 FOVs
    of 5 × 36 × 36, four rounds of four channels) and write the workflow inputs under ``WORK/input/``: one OME-TIFF
    per round, FOV and channel, the nuclear stain as the reference round's ``ch04.tif``, an additional round
    ``flamingo`` (``ch04.tif`` the stain, ``ch05.tif`` the reference round's channel maximum), these three as uint16, ``genes.csv``,
    ``fovs.csv`` and ``positions.csv``; the truth cell labels under ``WORK/labels/`` as plain ZYX TIFFs without
    stored metadata; ``WORK/fixture.json`` lists
    every file with its SHA-256. Run with the package's environment.
``run-migrated WORK --snakemake EXE`` (G3, W-737)
    The refactored workflow (``workflow/Snakefile``) on A to D migrated to the Python shape
    (``migrated_configurations``), in ``WORK/output/``, with one numerical thread: A (target ``all``: the ``fov``,
    ``morphology``, ``report`` and ``compare`` tasks), B (target the ``assign`` tasks' records, after the ``segment``
    tasks that import the truth labels), C (target the BigStitcher macro task's record), then D's three routes in
    their own output roots with target ``all``, the per-FOV route's fused cell labels written as the after-fusion
    route's import file in between. ``WORK/commands-g3.json`` records the commands and their exit status.
``extract WORK OUT [--migrated]``
    Write the expected values of the outputs of A to C to ``OUT`` (``values.json``, tables with ``dtypes.json``,
    ``arrays.npz``); with ``--migrated``, of ``run-migrated``'s, whose workflow writes none of the RETIRED_IMAGES.
    Reads the outputs with NumPy, pandas, tifffile and json only.
``extract-export WORK OUT`` (W-643)
    The expected values of invocation D in the layout of ``extract``: the records, tables and images of each route's
    ``stitching/``, ``export/`` and ``report/`` folders, the content of each ``sample.h5ad`` (X, layers, obsm, obs,
    var and the normalized uns) and the import file; its per-FOV outputs come from the configurations of A and B,
    which pin them. Run with an environment that has anndata.
``compare EXPECTED ACTUAL``
    Compare two ``extract`` (or two ``extract-export``) folders; print the differences and exit 1 when there is one.
``compare-migrated EXPECTED ACTUAL SCRATCH`` (G3, W-737)
    Compare the refactored workflow's values with an expected folder across environments, after the contract's
    exceptions (``compare_migrated``; the expected values with OC3 applied are written to ``SCRATCH``): print each
    difference with the decision that explains it, and exit 1 when one is unexplained.

The comparison classes of ``values.json`` (docs/contributing.md): ``exact`` (integers, counts, strings, statuses,
booleans, integer images, the fixture's file hashes), ``float64`` (rtol 1e-9, atol 1e-12), ``float32`` (rtol 1e-6,
atol 1e-6) and ``same_environment`` (the SHA-256 of an output file's bytes: identity in one environment, which a run
in another environment does not have to reproduce). Time, host, version and absolute-path fields are left out of the
records (``NORMALIZED``). A comparison across environments (``--other-environment``) also skips the fixture's file
hashes, the file digests of the stage records and the report's records, and the two output-file digests the per-FOV
records hold (``compare``).
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
from starfinder._execution import THREAD_VARIABLES  # noqa: E402
# R8: the comparison lives in the package's evaluation; this script records the goldens and calls it.
from starfinder.evaluation.workflow import (TOLERANCES, array_differences, record_differences,  # noqa: E402
                                            table_differences, value_differences, values_match as _close)

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


# --- the recorded configurations (W-642), from which G3's are migrated -----------------------------------------------

def configurations(work, starfinder_path):
    """(config A, config B, config C) of the recorded run: B is A with the stitching, assembly and export blocks,
    without the report block; C is A with a stitching block of method bigstitcher, without the report block. Their
    Python shape is ``migrated_configurations``'s."""
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


def _environment(work):
    """The environment of the Snakemake runs: the caller's, with one numerical thread whatever the caller sets.

    Each run record holds the thread variables in effect (``_execution.THREAD_VARIABLES``), compared exactly, and the
    expected records hold "1"; ITK's thread count is pinned with them, so the result does not depend on the caller.
    """
    threads = dict.fromkeys(THREAD_VARIABLES + ("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",), "1")
    return dict(os.environ, XDG_CACHE_HOME=str(Path(work) / "cache"), PYTHONDONTWRITEBYTECODE="1", **threads)


# --- invocation D: stitching, the export on each route and the report after the export (W-643) ----------------------

# The routes of export_sample the fixture runs without a learned method: per FOV (the saved assignments of
# reads_assignment), molecules only (assembly.assignment null) and after fusion (the fused projection segmented by an
# import run of the per-FOV route's fused cell labels). Each route runs in its own output root, golden-<route>.
EXPORT_ROUTES = ("per_fov", "molecules_only", "after_fusion")
# The per-FOV route's fused cell labels (export/<sample>/cell_labels.ome.tif, fuse_labels at the fusion's scale), copied
# as the label file that the route after fusion imports on the fused projection grid.
FUSED_CELLS = "fused-cells.ome.tif"
def export_output_root(work, route):
    return Path(work) / "output" / DATASET_ID / f"{OUTPUT_ID}-{route.replace('_', '-')}"


def export_configurations(work, starfinder_path):
    """{route: configuration} of the recorded invocation D: each is configuration A (its processing and morphology rules) with B's
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


# --- G3: the refactored workflow on the migrated configurations (W-737) ---------------------------------------------

# The invocations of the refactored workflow, in order: A to D migrated to the Python shape (design/specs/orchestration.md,
# "Migration from the current Python keys"). A2 has none: enhance_dapi_with_flamingo is retired (D6).
MIGRATED = ("A", "B", "C", "D-per-fov", "D-molecules-only", "D-after-fusion")
# The segmentation run of the migrated B and D-per-fov: the import of the truth cell labels, named as the legacy rule
# whose label file B copied them as.
LEGACY_RUN = "stardist_segmentation"


def migrated_configurations(work):
    """{invocation: configuration} of the refactored workflow (G3), each in the Python shape.

    A: configuration A's processing rule as the processing block (its parameters without the run flags, the checkpoint
    stages it states, candidates and pre_qc), its morphology rules as the morphology block with the channel images
    (``morphology.channel_images``, no projection: ``maximum_projection: false``), and the report block.
    B: A without the report block, with one segmentation run that imports each FOV's truth cell labels
    (``labels/{fov_id}.tif``, OP3; B copied them as the legacy label files of stardist_segmentation) and B's
    assignment, named ``default``, without expansion; the run is named ``stardist_segmentation`` (LEGACY_RUN). C: A without the report block, with C's stitching block
    (``img_row`` and ``img_col`` are removed, D6). D: invocation D's three routes, each A with D's stitching, assembly,
    export and report blocks, the per-FOV route with B's segmentation and assignment. ``rotate_nuclei``,
    ``create_nuclei_amplicon_overlay``, ``enhance_dapi_with_flamingo`` and ``stardist_segmentation`` are retired (D6).
    """
    work = Path(work)
    a, b, c = configurations(work, REPO)
    d = export_configurations(work, REPO)
    parameters = a["rules"]["rsf_single_fov"]["parameters"]
    processing = {k: v for k, v in json.loads(json.dumps(parameters)).items() if k != "checkpoints"}
    for block in processing.values():
        if isinstance(block, dict):
            block.pop("run", None)
    processing["checkpoints"] = list(parameters["checkpoints"]["stages"])
    removed = {"config_path", "starfinder_path", "workflow_mode", "rules", "maximum_projection", "img_row", "img_col"}
    base = {k: v for k, v in a.items() if k not in removed | {"report"}}
    base.update(processing=processing, morphology={"channel_images": {"projection": None}})
    # The run keeps the name of the legacy label file's rule, so that the label namespaces stay those of A to D.
    segmentation = {"runs": [{"name": LEGACY_RUN, "target": "cell", "method": "import",
                              "path": str(work / "labels" / "{fov_id}.tif")}]}
    assignment = {"cells": LEGACY_RUN, "name": ASSIGNMENT}
    out = {"A": dict(base, report={}),
           "B": dict(base, segmentation=segmentation, assignment=assignment),
           "C": dict(base, stitching=c["stitching"])}
    for route in EXPORT_ROUTES:
        config = dict(base, output_id=d[route]["output_id"], stitching=d[route]["stitching"],
                      assembly=d[route]["assembly"], export=d[route]["export"], report={})
        if route == "per_fov":
            config.update(segmentation=segmentation, assignment=assignment)
        out["D-" + route.replace("_", "-")] = config
    if b["assembly"]["assignment"] != ASSIGNMENT or tuple(out) != MIGRATED:
        raise ValueError("the migrated configurations do not follow A to D")
    return out


def run_migrated(work, snakemake, starfinder_path=REPO, cores=1, names=MIGRATED):
    """G3's runs: the refactored workflow (``workflow/Snakefile``, ``backend: python``) on the migrated configurations
    ``names`` (a subset of MIGRATED) in order, with one numerical thread (``_environment``) and the ``starfinder``
    command beside ``snakemake`` first on PATH. A, the D routes: target ``all``; B: the assign tasks' records; C: the
    BigStitcher macro task's record (its engine task runs Fiji). The per-FOV route's fused cell labels are written as
    the after-fusion route's import file before the next route, as the recorded invocation D did. Records the commands
    in ``WORK/commands-g3.json``; returns them."""
    import yaml
    from starfinder import _layout
    work = Path(work)
    fov_ids = json.loads((work / "fixture.json").read_text())["fov_ids"]
    configs = migrated_configurations(work)
    paths = {}
    for name, config in configs.items():
        paths[name] = work / f"config-g3-{name.lower()}.yaml"
        paths[name].write_text(yaml.safe_dump(config, sort_keys=False))
    snakefile = Path(starfinder_path) / "workflow" / "Snakefile"
    environment = _environment(work)
    environment["PATH"] = os.pathsep.join([str(Path(snakemake).parent), environment.get("PATH", "")])
    common = [str(snakemake), "-s", str(snakefile), "--directory", str(work / "snakemake"), "--cores", str(cores),
              "--nocolor", "--printshellcmds"]
    out = output_root(work)
    targets = {"B": [_layout.task_record(out, f"assign/{SAMPLE_ID}/{fov}/{ASSIGNMENT}") for fov in fov_ids],
               "C": [_layout.task_record(out, f"bigstitcher_macro/{SAMPLE_ID}")]}
    records = []
    for name in (n for n in MIGRATED if n in names):
        if name == "D-molecules-only" or (name == "D-after-fusion" and "D-molecules-only" not in names):
            _fused_cells(export_output_root(work, "per_fov") / "export" / SAMPLE_ID, work / "labels" / FUSED_CELLS)
        log = work / f"snakemake-g3-{name}.log"
        command = common + ["--configfile", str(paths[name]), "--", *map(str, targets.get(name, ["all"]))]
        with open(log, "w") as handle:
            status = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=environment).returncode
        records.append({"name": name, "command": command, "exit_status": status, "log": log.name})
        (work / "commands-g3.json").write_text(json.dumps(records, indent=1) + "\n")
        if status:
            raise SystemExit(f"invocation {name} exited with status {status}; see {log}")
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


# The image folders of rules that the refactored workflow retires (D6): G3's extraction does not read them.
RETIRED_IMAGES = ("DAPI", "overlay", "flamingo/enhanced_DAPI")


def extract(work, out, *, migrated=False):
    """The expected values of one run in ``out``: ``records/`` (each JSON record normalized), the tables with their
    ``dtypes.json`` (the report's intensity histograms summarized, one row per histogram), ``arrays.npz`` (each FOV's
    reference image) and ``values.json`` (the fixture's file digests; the shape, dtype, minimum, maximum and sum of
    the other images and of the BigStitcher tiles; the macro's text; and the output files' digests). ``migrated``: the
    output of ``run-migrated`` (G3), whose workflow writes none of the RETIRED_IMAGES."""
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
            if migrated and folder_name in RETIRED_IMAGES:
                continue
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

# The digest a run record holds of each checkpoint stage record: candidates.json names the software it ran with
# (detection_diagnostics.software), so the digest is an identity of the recording environment, like the file digests,
# and a comparison across environments skips it (W-643: NumPy 2.3.5 against 2.2.6 changed only that leaf).
STAGE_RECORD_DIGEST = re.compile(r"^records/fovs/[^/]+/run\.json\.checkpoints\.[A-Za-z_]+\.sha256: ")
# The fixture's file digests (values.json's fixture: keys) and the digests the report records of the files it wrote
# (report.json's files[<n>].sha256, under report/<sample>/ at any depth of the records): file digests hold only in the
# recording environment (design/validation/orchestration.md, G1's row), and these change with NumPy's SIMD dispatch
# (W-751: AVX-512 off and OPENBLAS_CORETYPE=Zen changed only them), so a comparison across environments skips them.
FIXTURE_DIGEST = "fixture:"
REPORT_FILE_DIGEST = re.compile(r"^records/(?:[^/]+/)*report/[^/]+/report\.json\.files\[\d+\]\.sha256: ")
# The output-file digests held inside the per-FOV records: the pre_qc stage record's digest of its table and run.json's
# digest of molecules.csv. The expected values limit output-file digests to the recording environment (the core
# values' README.md), and the molecule table is compared with float tolerances, so its bytes need not match across
# CPUs (W-751 found these values changed under a reduced SIMD dispatch; operator decision, 2026-10-30). Matched exactly.
OUTPUT_FILE_DIGEST = re.compile(r"^records/fovs/[^/]+/(?:checkpoints/pre_qc\.json\.files\[0\]|"
                                r"run\.json\.outputs\.molecules)\.sha256: ")


def compare(expected_dir, actual_dir, *, same_environment=True):
    """The differences between two ``extract`` folders, as readable lines (empty when they match)."""
    expected_dir, actual_dir = Path(expected_dir), Path(actual_dir)
    lines = value_differences(json.loads((expected_dir / "values.json").read_text()),
                              json.loads((actual_dir / "values.json").read_text()),
                              same_environment=same_environment, environment_keys=(FIXTURE_DIGEST,))
    names = lambda folder: {p.relative_to(folder / "records").as_posix() for p in (folder / "records").rglob("*.json")}
    for name in sorted(names(expected_dir) | names(actual_dir)):
        e, a = expected_dir / "records" / name, actual_dir / "records" / name
        if not (e.is_file() and a.is_file()):
            lines.append(f"records/{name}: only in {'expected' if e.is_file() else 'actual'}")
            continue
        lines += [f"records/{name}{line}" for line in record_differences(json.loads(a.read_text()),
                                                                          json.loads(e.read_text()))
                  if same_environment or not (STAGE_RECORD_DIGEST.match(f"records/{name}{line}")
                                              or REPORT_FILE_DIGEST.match(f"records/{name}{line}")
                                              or OUTPUT_FILE_DIGEST.match(f"records/{name}{line}"))]
    dtypes_e = json.loads((expected_dir / "dtypes.json").read_text())
    dtypes_a = json.loads((actual_dir / "dtypes.json").read_text())
    for name in sorted(set(dtypes_e) | set(dtypes_a)):
        if name not in dtypes_a or name not in dtypes_e:
            lines.append(f"{name}: only in {'expected' if name in dtypes_e else 'actual'}")
            continue
        e, a = EV.read_table(expected_dir / name), EV.read_table(actual_dir / name)
        close = [c for c, d in dtypes_e[name].items() if d == "float64" and not _integral(e[c])]
        lines += [f"{name}: {line}" for line in table_differences(a, e, close)]
    # A float array (an AnnData matrix, repair 1) within the tolerance of its dtype, NaN equal to NaN.
    lines += array_differences(EV.read_arrays(expected_dir / "arrays.npz"), EV.read_arrays(actual_dir / "arrays.npz"))
    return lines


# --- G3's comparison: the contract's exceptions (design/specs/orchestration.md, "The exception of the golden
# comparison") ---------------------------------------------------------------------------------------------------

# OC3 on the expected values: rotate_angle 0 is no rotation, so run.json and the report's step tables have no
# rotate_round step (the later steps' indices shift, in the step tables and in the row keys that name them), every
# frame id loses /rotate:0 and the pipeline's rotation_degrees is null.
ROTATION_STEP = "rotate_round"
ROW_KEY = re.compile(r"scope=fov;fov_id=([^;]+);index=(\d+)")
_RECORD = r"^records/(?:(?:per_fov|molecules_only|after_fusion)/)?"
_FOV = r"fovs/FOV_\d+/"
# Each difference that the exceptions leave, by decision: (decision, pattern of the comparison's line, why).
EXCEPTIONS = (
    ("OC3", _RECORD + _FOV + r"run\.json\.outputs\.reference_image\.sha256: ",
     "the reference image's digest: its stored metadata holds the frame id"),
    ("OC3", _RECORD + r"stitching/[^/]+/placement\.json\.inputs\.FOV_\d+\.file_sha256: ",
     "the placement's digest of each FOV's reference image"),
    ("OC3", _RECORD + r"export/[^/]+/export\.json\.fusion\.fovs\[\d+\]\.images\.reference_merged\.sha256: ",
     "the export's digest of each fused reference image"),
    ("OC3", _RECORD + r"(?:" + _FOV + r"assignment/[^/]+/assignment\.json\.inputs\.cells\.file|"
     r"export/[^/]+/(?:export\.json|sample\.h5ad\.uns\.json)\.(?:labels\.fovs\[\d+\]|assembly\.fovs\[\d+\]\.territories))"
     r"\.sha256: ",
     "the digest of each FOV's cell label image (the run's labels themselves, labels_sha256, are compared)"),
    ("D6", r"^images/(?:DAPI|overlay|flamingo/enhanced_DAPI)/FOV_\d+\.tif:(?:shape|dtype|min|max|sum): only in expected$",
     "the images of the retired rules rotate_nuclei, create_nuclei_amplicon_overlay and enhance_dapi_with_flamingo"),
    ("D6", r"^file:images/(?:DAPI|overlay|flamingo/enhanced_DAPI|stardist_segmentation)/FOV_\d+\.tif: only in expected$",
     "the files of the retired rules, and the legacy label file of stardist_segmentation that B copied the truth "
     "labels as (the segment task imports them, OP3)"),
    ("D6", r"^file:" + _FOV + r"assignment/[^/]+/cell_labels\.tif: only in expected$|"
     + _RECORD + _FOV + r"assignment/[^/]+/assignment\.json\.files\.cell_labels: only in expected$",
     "the assignment folder's copy of the cell labels: the assign task reads the segment task's saved run and "
     "references its labels instead of copying them"),
    ("D6", _RECORD + _FOV + r"assignment/[^/]+/assignment\.json\.(?:workflow: only in expected$|"
     r"inputs\.cells\.saved_under_run: True != False|"
     r"inputs\.cells\.file\.path: '\.\./\.\./segmentation/[^/']+/labels\.tif' != 'cell_labels\.tif'|"
     r"inputs\.cells\.record\.import\.path: '<work>/labels/(?P<fov>FOV_\d+)\.tif' != "
     r"'<work>/output/[^']+/images/stardist_segmentation/(?P=fov)\.tif'|"
     r"inputs\.cells\.record\.(?:upstream|labels\.path|labels\.file_sha256): only in actual$)",
     "the cell run's source: the segment task's saved run (saved under the run, with its record's upstream and label "
     "file) instead of reads_assignment's legacy label file, and the rule parameters' record of reads_assignment "
     "(rules.reads_assignment.parameters, retired)"),
    ("D6", _RECORD + r"export/[^/]+/(?:export\.json|sample\.h5ad\.uns\.json)\.(?:labels\.fovs\[\d+\]|"
     r"assembly\.fovs\[\d+\]\.territories)\.path: '<work>/output/[^']+/" + _FOV
     + r"segmentation/[^/']+/labels\.tif' != '<work>/output/[^']+/" + _FOV + r"assignment/[^/']+/cell_labels\.tif'",
     "the export's path of each FOV's cell labels: the segment task's saved run"),
    ("D6", r"^file:" + _FOV + r"(?:other_rounds|segmentation)/.+: only in actual$",
     "the outputs of the morphology task's saved forms and of the segment task, which no rule of A to D wrote"),
    ("D8", r"^file:workflow/.+: only in actual$", "the plan's files, the task records and the task logs"),
    ("D8", _RECORD + _FOV + r"run\.json\.workflow: only in actual$", "run.json's workflow section"),
)


def _oc3(value, indices=None):
    """OC3 on a record: no rotate_round step, no /rotate:0 in a frame id, a null rotation_degrees for 0, and the row
    keys of FOV steps renumbered by ``indices`` ({fov: {old index: new index}})."""
    if isinstance(value, dict):
        out = {k: _oc3(v, indices) for k, v in value.items()}
        if out.get("rotation_degrees") == 0:
            out["rotation_degrees"] = None
        return out
    if isinstance(value, list):
        return [_oc3(v, indices) for v in value if not (isinstance(v, dict) and v.get("name") == ROTATION_STEP)]
    if isinstance(value, str):
        value = value.replace("/rotate:0", "")
        if indices:
            value = ROW_KEY.sub(lambda m: f"scope=fov;fov_id={m[1]};index={indices[m[1]][int(m[2])]}"
                                if m[1] in indices and int(m[2]) in indices[m[1]] else m[0], value)
        return value
    return value


def expected_with_exceptions(expected_dir, target):
    """A copy of an expected folder (W-642's or invocation D's) with OC3 applied: the rotate_round rows leave the
    report's step tables, whose indices are renumbered within each FOV, and the row keys that name a step index follow
    them in the tables and records; the records lose their rotate_round steps and /rotate:0."""
    target = Path(target)
    shutil.copytree(expected_dir, target)
    dtypes = json.loads((target / "dtypes.json").read_text())
    indices = {}
    for name in sorted(dtypes):
        if not name.endswith("-tables-steps.csv"):
            continue
        frame = EV.read_table(target / name)
        kept = frame[frame["name"] != ROTATION_STEP].copy()
        route = name[:-len("report-sample-tables-steps.csv")]
        for fov, rows in kept[kept["scope"] == "fov"].groupby("fov_id", sort=False):
            indices.setdefault(route, {})[fov] = dict(zip(rows["index"].tolist(), range(len(rows))))
        renumbered = kept["scope"] == "fov"
        kept.loc[renumbered, "index"] = kept[renumbered].groupby("fov_id", sort=False).cumcount().astype("int64")
        dtypes[name] = EV.write_csv(kept.reset_index(drop=True), target / name)
    for name in sorted(dtypes):
        if name.endswith("-tables-unavailable.csv"):
            frame = EV.read_table(target / name)
            route = indices.get(name[:-len("report-sample-tables-unavailable.csv")], {})
            frame["row_key"] = pd.array([_oc3(v, route) if isinstance(v, str) else v for v in frame["row_key"]],
                                        dtype="string")
            dtypes[name] = EV.write_csv(frame, target / name)
    (target / "dtypes.json").write_text(json.dumps(dtypes, indent=1) + "\n")
    for path in sorted((target / "records").rglob("*.json")):
        parts = path.relative_to(target / "records").parts
        route = indices.get(f"{parts[0]}-" if parts[0] in EXPORT_ROUTES else "", {})
        path.write_text(json.dumps(_oc3(json.loads(path.read_text()), route), indent=1) + "\n")
    return target


# The report's inputs (report.json's ``inputs``), compared entry by entry by ``_report_input_differences``, never by
# EXCEPTIONS: the label entries of D6 (the expected copy of each FOV's labels that reads_assignment wrote, against the
# segment task's saved run) and, of every other entry compared field by field in order, OC3 only for the digest and
# size of a reference image.
REPORT_INPUTS = re.compile(r"^records/(.+?/report\.json)\.inputs(?=[\[.:])")
EXPECTED_LABELS = re.compile(r"^fovs/(FOV_\d+)/assignment/[^/]+/cell_labels\.tif$")
ACTUAL_LABELS = {re.compile(r"^fovs/(FOV_\d+)/segmentation/[^/]+/labels\.tif$"): "label_image",
                 re.compile(r"^fovs/(FOV_\d+)/segmentation/[^/]+/segmentation\.json$"): "segmentation_record"}


def _report_input_differences(name, expected, actual):
    """[(decision or None, line)] of the ``inputs`` of one report record (``records/<name>``) in two folders.

    The label entries leave both lists first: the expected ``fovs/<fov>/assignment/<name>/cell_labels.tif``, the actual
    ``fovs/<fov>/segmentation/<run>/labels.tif`` (``label_image``, verified) and ``segmentation.json``
    (``segmentation_record``, verified), one D6 line each; the FOVs of the two label files must be the same. The other
    entries are compared in order with ``record_differences``; a ``sha256`` or ``bytes`` line is OC3 only when that
    expected entry's path ends in ``/reference.tif``. Every other line is unexplained."""
    where = f"records/{name}.inputs"
    if not isinstance(expected, list) or not isinstance(actual, list):
        return [(None, f"{where}: {'missing' if not isinstance(actual, list) else 'not'} in actual, "
                       f"{'missing' if not isinstance(expected, list) else 'present'} in expected")]
    out, kept_expected, kept_actual, removed = [], [], [], {"expected": set(), "actual": set()}
    for index, entry in enumerate(expected):
        match = EXPECTED_LABELS.match(entry.get("path") or "")
        if match:
            removed["expected"].add(match[1])
            out.append(("D6", f"{where}[{index}]: {entry['path']} only in expected (the copied labels, D6)"))
        else:
            kept_expected.append((index, entry))
    for index, entry in enumerate(actual):
        path = entry.get("path") or ""
        match, kind = next(((pattern.match(path), k) for pattern, k in ACTUAL_LABELS.items() if pattern.match(path)),
                           (None, None))
        if match is None:
            kept_actual.append(entry)
            continue
        if kind == "label_image":
            removed["actual"].add(match[1])
        if entry.get("kind") == kind and entry.get("verified") is True:
            out.append(("D6", f"{where}[{index}]: {path} only in actual (the segment task's saved run, D6)"))
        else:
            out.append((None, f"{where}[{index}]: {path} only in actual with kind {entry.get('kind')!r} and verified "
                              f"{entry.get('verified')!r}, not {kind!r} and True"))
    for fov in sorted(removed["expected"] ^ removed["actual"]):
        side = "a copied cell_labels.tif and no labels.tif" if fov in removed["expected"] else \
            "a labels.tif and no copied cell_labels.tif"
        out.append((None, f"{where}: FOV {fov} has {side}"))
    if len(kept_expected) != len(kept_actual):
        return out + [(None, f"{where}: {len(kept_actual)} entries besides the label entries != {len(kept_expected)}")]
    for (index, e), a in zip(kept_expected, kept_actual):
        for line in record_differences(a, e, f"{where}[{index}]"):
            oc3 = re.match(rf"^{re.escape(where)}\[\d+\]\.(?:sha256|bytes): ", line) and \
                (e.get("path") or "").endswith("/reference.tif")
            out.append(("OC3" if oc3 else None, line))
    return out


def compare_migrated(expected_dir, actual_dir, target):
    """G3: the differences between an expected folder and the refactored workflow's values (``extract`` with
    ``migrated``, or ``extract_export``), across environments, after the contract's exceptions.

    The expected values are copied to ``target`` with OC3 applied (``expected_with_exceptions``). The report records'
    ``inputs`` are taken out of ``compare``'s lines and compared entry by entry (``_report_input_differences``); each
    other difference is attributed to the first of EXCEPTIONS whose pattern it matches. Returns (the unexplained
    differences, {decision: the differences it explains})."""
    expected = expected_with_exceptions(expected_dir, target)
    explained, unexplained = {}, []
    found = []
    reports = set()
    for line in compare(expected, actual_dir, same_environment=False):
        report = REPORT_INPUTS.match(line)
        if report:
            reports.add(report[1])
            continue
        found.append((next((d for d, pattern, _ in EXCEPTIONS if re.search(pattern, line)), None), line))
    for name in sorted(reports):
        records = [json.loads((Path(folder) / "records" / name).read_text()) for folder in (expected, actual_dir)]
        found += _report_input_differences(name, records[0].get("inputs"), records[1].get("inputs"))
    for decision, line in found:
        if decision is None:
            unexplained.append(line)
        else:
            explained.setdefault(decision, []).append(line)
    return unexplained, explained


def _integral(series):
    """Whether a float column holds integer values only (compared exactly, docs/contributing.md)."""
    values = series.to_numpy(np.float64)
    finite = np.isfinite(values)
    return bool(np.array_equal(values[finite], np.round(values[finite])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fixture").add_argument("work", type=Path)
    run = commands.add_parser("run-migrated")
    run.add_argument("work", type=Path)
    run.add_argument("--snakemake", required=True)
    run.add_argument("--starfinder-path", type=Path, default=REPO)
    run.add_argument("--cores", type=int, default=1)
    extracting = commands.add_parser("extract")
    extracting.add_argument("work", type=Path)
    extracting.add_argument("out", type=Path)
    extracting.add_argument("--migrated", action="store_true", help="the output of run-migrated (G3)")
    extracting_d = commands.add_parser("extract-export")
    extracting_d.add_argument("work", type=Path)
    extracting_d.add_argument("out", type=Path)
    comparing = commands.add_parser("compare")
    comparing.add_argument("expected", type=Path)
    comparing.add_argument("actual", type=Path)
    comparing.add_argument("--other-environment", action="store_true",
                           help="skip the file digests: the same_environment values, the fixture's, "
                                "the stage records', the report files' and the per-FOV records' digests of "
                                "pre_qc.csv and molecules.csv")
    migrated = commands.add_parser("compare-migrated")
    migrated.add_argument("expected", type=Path)
    migrated.add_argument("actual", type=Path)
    migrated.add_argument("scratch", type=Path, help="a new folder for the expected values with OC3 applied")
    args = parser.parse_args(argv)
    if args.command == "fixture":
        record = write_fixture(args.work)
        print(f"{len(record['files'])} fixture files in {args.work}")
    elif args.command == "run-migrated":
        for record in run_migrated(args.work, args.snakemake, args.starfinder_path, args.cores):
            print(record["name"], record["exit_status"])
    elif args.command == "extract":
        print(f"{len(extract(args.work, args.out, migrated=args.migrated))} values in {args.out}")
    elif args.command == "extract-export":
        print(f"{len(extract_export(args.work, args.out))} values in {args.out}")
    elif args.command == "compare-migrated":
        unexplained, explained = compare_migrated(args.expected, args.actual, args.scratch)
        for decision, lines in explained.items():
            print("\n".join(f"{decision}: {line}" for line in lines))
        print("\n".join(f"unexplained: {line}" for line in unexplained) if unexplained else "no unexplained difference")
        return 1 if unexplained else 0
    else:
        lines = compare(args.expected, args.actual, same_environment=not args.other_environment)
        print("\n".join(lines) if lines else "no difference")
        return 1 if lines else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
