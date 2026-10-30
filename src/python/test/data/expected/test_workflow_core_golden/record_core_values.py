"""Record and compare the per-FOV core values of the current workflow on the session dataset (W-734, X4).

The values that the refactored ``fov`` task must reproduce on the pytest session's small synthetic dataset
(design/validation/orchestration.md, X4): today's ``rsf_single_fov`` rule (``backend: python``) run by Snakemake 9 on
each FOV, and the explicit expected values of what it writes, in the layout of ``scripts/workflow_golden.py``: the
molecule table, ``run.json`` (its counts and stage records), the reference image and the checkpoint stages.

Subcommands, each run in its own process, from ``src/python`` with ``PYTHONPATH`` naming this checkout's ``src/python``:

``fixture WORK``
    Generate the session dataset as ``test/conftest.py`` (``small_dataset``) does, with the same command line
    (``SMALL_DATASET_ARGS``: preset ``small``, seed 42; two FOVs of 16 × 256 × 256, four rounds of four channels,
    uint16), into the input root ``WORK/input/session/small/``; add the workflow's ``fovs.csv`` beside it. Writes
    ``WORK/fixture.json`` with every input file's SHA-256 (``manifest.json`` of the generator, which records its
    absolute output folder, is listed without one).
``run WORK --snakemake EXE``
    Write ``WORK/config.yaml`` and run ``rsf_single_fov`` of each FOV in ``WORK/output/session/core/`` (the target
    ``molecules.csv`` of each FOV, the one rule allowed, ``--cores 1``). ``WORK/commands.json`` records the command
    and its exit status.
``extract WORK OUT``
    Write the core values to ``OUT``: ``records/`` (each FOV's ``run.json`` and checkpoint stage records,
    normalized as ``workflow_golden`` normalizes records), the tables with ``dtypes.json`` (each FOV's
    ``molecules.csv`` and checkpoint stage tables), ``arrays.npz`` (each FOV's reference image) and ``values.json``
    (the fixture's digests, exact, and the digests of the output files that hold no absolute path,
    ``same_environment``; ``VARYING``).
``compare EXPECTED ACTUAL [--other-environment]``
    ``workflow_golden.compare`` of two ``extract`` folders; prints the differences and exits 1 when there is one.

The comparison classes are those of ``workflow_golden`` (docs/contributing.md, "Expected values and tolerances").
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PYTHON = HERE.parents[3]
REPO = PYTHON.parents[1]
sys.path.insert(0, str(PYTHON / "scripts"))
sys.path.insert(0, str(PYTHON / "test"))
import expected_values as EV  # noqa: E402
import workflow_golden as G  # noqa: E402

DATASET_ID = "session"
SAMPLE_ID = "small"
OUTPUT_ID = "core"
ROUNDS = ("round1", "round2", "round3", "round4")
CHANNELS = ("ch00", "ch01", "ch02", "ch03")
RULE = "rsf_single_fov"
# The generator's manifest records the absolute folder it wrote: listed in fixture.json without a digest.
FIXTURE_VARYING = ("manifest.json",)
# Besides workflow_golden.VARYING: the outputs whose bytes hold the frame id of the reference round, which names the
# absolute input folder (the session dataset's TIFFs store no frame; the contract's OC4): the reference image and the
# candidates stage record. Two recordings in different folders differ in nothing else. Their digests are left out,
# wherever a record lists them too; their content is kept (the image whole, the record normalized).
VARYING = G.VARYING + ("reference.tif", "checkpoints/candidates.json")


def input_root(work):
    return Path(work) / "input" / DATASET_ID / SAMPLE_ID


def output_root(work):
    return Path(work) / "output" / DATASET_ID / OUTPUT_ID


# --- fixture ------------------------------------------------------------------------------------------------------

def write_fixture(work):
    """The session dataset and the workflow's fovs.csv under ``work`` (a new or empty folder)."""
    from conftest import SMALL_DATASET_ARGS, SMALL_DATASET_GENERATOR
    from starfinder.__main__ import main as cli
    work = Path(work)
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"{work} is not empty")
    root = input_root(work)
    status = cli(["synthetic", "generate", *SMALL_DATASET_ARGS, "--owner", "pytest", "--output", str(root)])
    if status != 0:
        raise SystemExit(f"synthetic generation exited with {status}")
    generation = json.loads((root / "generation.json").read_text())
    if {k: generation[k] for k in SMALL_DATASET_GENERATOR} != SMALL_DATASET_GENERATOR:
        raise ValueError("the generator is not the session dataset's")
    fov_ids = list(generation["fov_ids"])
    (root / "fovs.csv").write_text("sample,fov_id\n" + "".join(f"{SAMPLE_ID},{fov}\n" for fov in fov_ids))
    files = {}
    for path in sorted(p for p in work.rglob("*") if p.is_file()):
        name = path.relative_to(work).as_posix()
        files[name] = {"bytes": path.stat().st_size}
        if path.name not in FIXTURE_VARYING:
            files[name]["sha256"] = G.sha256_file(path)
    # shape_zyx: workflow_golden.configurations reads it (its configuration C).
    record = {"dataset": "test/conftest.py small_dataset", "args": list(SMALL_DATASET_ARGS),
              "generator": SMALL_DATASET_GENERATOR, "fov_ids": fov_ids,
              "shape_zyx": list(generation["provenance"][fov_ids[0]]["requested_config"]["shape_zyx"]),
              "rounds": list(ROUNDS), "channels": list(CHANNELS), "files": files}
    (work / "fixture.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


# --- the configuration and the run --------------------------------------------------------------------------------

def configuration(work, starfinder_path):
    """The workflow configuration of the run: the Python backend, the session dataset's layout ({fov}/{round}) and
    codebook, and the per-FOV core of rsf_single_fov with W-642's processing parameters (golden configuration A,
    ``workflow_golden.configurations``) and its checkpoint stages."""
    work = Path(work)
    golden, _, _ = G.configurations(work, starfinder_path)
    rules = {RULE: json.loads(json.dumps(golden["rules"][RULE]))}
    return {
        "starfinder_path": str(starfinder_path), "config_path": str(work / "config.yaml"),
        "root_input_path": str(work / "input"), "root_output_path": str(work / "output"),
        "dataset_id": DATASET_ID, "sample_id": SAMPLE_ID, "output_id": OUTPUT_ID, "backend": "python",
        "workflow_mode": "free", "n_rounds": len(ROUNDS), "ref_round": ROUNDS[0], "rotate_angle": 0,
        "maximum_projection": False, "voxel_size_xy": 0.5, "voxel_size_z": 1.0, "seq_channel_order": list(CHANNELS),
        # common.smk reads additional_round at parse time: the session dataset has none.
        "additional_round": [], "image_dir_template": "{fov}/{round}", "codebook": "codebook.csv", "rules": rules}


def run_workflow(work, snakemake, starfinder_path, cores=1):
    """rsf_single_fov of each FOV (the module docstring); returns the command records."""
    import yaml
    work = Path(work)
    fov_ids = json.loads((work / "fixture.json").read_text())["fov_ids"]
    config = configuration(work, starfinder_path)
    Path(config["config_path"]).write_text(yaml.safe_dump(config, sort_keys=False))
    snakefile = Path(starfinder_path) / "workflow" / "Snakefile"
    targets = [str(output_root(work) / "fovs" / fov / "molecules.csv") for fov in fov_ids]
    # --configfile and --allowed-rules take several values, so "--" ends them before the targets.
    command = [str(snakemake), "-s", str(snakefile), "--directory", str(work / "snakemake"), "--cores", str(cores),
               "--nocolor", "--printshellcmds", "--configfile", config["config_path"], "--allowed-rules", RULE,
               "--", *targets]
    environment = dict(os.environ, XDG_CACHE_HOME=str(work / "cache"), PYTHONDONTWRITEBYTECODE="1")
    log = work / "snakemake.log"
    with open(log, "w") as handle:
        status = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=environment).returncode
    records = [{"name": RULE, "command": command, "exit_status": status, "log": log.name}]
    (work / "commands.json").write_text(json.dumps(records, indent=1) + "\n")
    if status:
        raise SystemExit(f"the workflow exited with status {status}; see {log}")
    return records


# --- the core values ----------------------------------------------------------------------------------------------

def extract(work, out):
    """The core values of one run in ``out``: ``records/`` (each FOV's run.json and checkpoint stage records,
    normalized by ``workflow_golden._normalized`` with VARYING), the tables with their ``dtypes.json`` (each FOV's
    molecule table and checkpoint stage tables, read with the dtypes their records state), ``arrays.npz`` (each FOV's
    reference image) and ``values.json`` (the fixture's file digests, exact; the reference image's shape and dtype,
    exact; and the SHA-256 of each output file that holds no time or absolute path, those not in VARYING,
    ``same_environment``)."""
    work, out = Path(work).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fixture = json.loads((work / "fixture.json").read_text())
    root = output_root(work)
    values, dtypes, arrays = {}, {}, {}
    for name, entry in fixture["files"].items():
        if "sha256" in entry:
            values[f"fixture:{name}"] = {"value": entry["sha256"], "compare": "exact"}
    for fov in fixture["fov_ids"]:
        folder = root / "fovs" / fov
        run = json.loads((folder / "run.json").read_text())
        if run["status"] != "succeeded":
            raise ValueError(f"{folder / 'run.json'} says {run['status']!r}")
        stages = [folder / run["checkpoint_directory"] / Path(entry["path"]).name
                  for entry in run["checkpoints"].values()]
        for path in [folder / "run.json", *stages]:
            target = out / "records" / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            record = G._normalized(json.loads(path.read_text()), str(work), VARYING)
            target.write_text(json.dumps(record, indent=1) + "\n")
        tables = {f"{fov}-molecules.csv": (folder / "molecules.csv", run["outputs"]["molecules"]["dtypes"])}
        for path in stages:
            stage = json.loads(path.read_text())
            tables[f"{fov}-checkpoints-{stage['stage']}.csv"] = (path.parent / stage["table"], stage["dtypes"])
        # Read with the dtypes their records state, in the CSV convention of expected_values, which the checkpoint
        # CSV shares.
        for name, (path, types) in tables.items():
            dtypes[name] = EV.write_csv(EV.read_csv(path, types), out / name)
        reference = G._image(folder / "reference.tif")
        arrays[f"{fov}/reference"] = reference
        values[f"{fov}/reference:shape"] = {"value": list(reference.shape), "compare": "exact"}
        values[f"{fov}/reference:dtype"] = {"value": str(reference.dtype), "compare": "exact"}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if not G._varies(relative, VARYING):
            values[f"file:{relative}"] = {"value": G.sha256_file(path), "compare": "same_environment"}
    (out / "values.json").write_text(json.dumps(values, indent=1, sort_keys=True) + "\n")
    # Each table's columns keep their order (expected_values.read_csv reads them in it).
    (out / "dtypes.json").write_text(json.dumps({name: dtypes[name] for name in sorted(dtypes)}, indent=1) + "\n")
    np.savez_compressed(out / "arrays.npz", **arrays)
    return values


def compare(expected_dir, actual_dir, *, same_environment=True):
    """The differences between two ``extract`` folders, as readable lines (empty when they match):
    ``workflow_golden.compare``, whose classes and normalization these values share."""
    return G.compare(expected_dir, actual_dir, same_environment=same_environment)


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
    comparing = commands.add_parser("compare")
    comparing.add_argument("expected", type=Path)
    comparing.add_argument("actual", type=Path)
    comparing.add_argument("--other-environment", action="store_true",
                           help="skip the same_environment values (the file digests) and the stage-record digests")
    args = parser.parse_args(argv)
    if args.command == "fixture":
        record = write_fixture(args.work)
        print(f"{len(record['files'])} fixture files in {args.work}")
    elif args.command == "run":
        for record in run_workflow(args.work, args.snakemake, args.starfinder_path, args.cores):
            print(record["name"], record["exit_status"])
    elif args.command == "extract":
        print(f"{len(extract(args.work, args.out))} values in {args.out}")
    else:
        lines = compare(args.expected, args.actual, same_environment=not args.other_environment)
        print("\n".join(lines) if lines else "no difference")
        return 1 if lines else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
