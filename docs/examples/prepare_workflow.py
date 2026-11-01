"""Write a run configuration from one of the workflow examples, outside the checkout.

The example's keys stay as written; only root_input_path and root_output_path change, to
the input tree and to RUN/output. With --input-root the run reads that tree, for example
the synthetic fixture that ``scripts/workflow_golden.py fixture WORK`` writes under
WORK/input. Without it, a placeholder input tree is written under RUN/input: fovs.csv
with one FOV, a codebook row, a positions table and a blank 1 x 8 x 8 image for each
image and label file the plan looks for. The plan accepts it; it is not data to run.
Run from src/python, for example:

    uv run python ../../docs/examples/prepare_workflow.py minimal RUN --input-root WORK/input
    uv run starfinder workflow plan --config RUN/config.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import yaml

TEMPLATES = ("minimal", "full", "recipe-2")
PLACEHOLDER_FOV = "FOV_001"


def _label(entry):
    return entry if isinstance(entry, str) else entry["channel"]


def placeholder_inputs(config, root):
    """The placeholder input tree of ``config`` under ``root`` (the input root): plan only."""
    folder = root / config["dataset_id"] / config["sample_id"]
    template = config.get("image_dir_template", "{round}/{fov}")
    images = {f"round{i}": config["seq_channel_order"] for i in range(1, config["n_rounds"] + 1)}
    images[config["ref_round"]] = [*images[config["ref_round"]], *(config.get("reference_stains") or [])]
    for entry in config.get("additional_round") or []:
        images[entry["round_name"]] = entry.get("channel_order") or []
    for round_name, channels in images.items():
        directory = folder / template.format(round=round_name, fov=PLACEHOLDER_FOV)
        directory.mkdir(parents=True, exist_ok=True)
        for entry in channels:
            tifffile.imwrite(directory / f"{_label(entry)}.tif", np.zeros((1, 8, 8), np.uint16))
    (folder / "fovs.csv").write_text(f"sample,fov_id\n{config['sample_id']},{PLACEHOLDER_FOV}\n")
    (folder / "genes.csv").write_text("ExampleGene,CACCC\n")
    for run in (config.get("segmentation") or {}).get("runs") or []:
        if run.get("method") == "import" and not Path(run["path"]).is_absolute():
            labels = folder / run["path"].replace("{fov_id}", PLACEHOLDER_FOV)
            labels.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(labels, np.zeros((1, 8, 8), np.uint16))
    positions = ((config.get("stitching") or {}).get("positions") or {}).get("table")
    if positions and not Path(positions).is_absolute():
        (folder / positions).write_text(f"fov_id,x,y\n{PLACEHOLDER_FOV},0,0\n")
    (folder / "PLAN_ONLY.txt").write_text("Blank placeholder images for the plan; not data to run.\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("template", choices=TEMPLATES, help="the example: workflow-<template>.yaml")
    parser.add_argument("run", type=Path, help="a new directory outside the checkout")
    parser.add_argument("--input-root", type=Path,
                        help="the input tree's root_input_path; default a placeholder tree under RUN/input")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    run = args.run.resolve()
    if run == repo or repo in run.parents:
        parser.error("RUN must be outside the checkout")
    if run.exists():
        parser.error("RUN must be a new directory (existing runs are kept)")
    source = Path(__file__).with_name(f"workflow-{args.template}.yaml")
    text = source.read_text()
    config = yaml.safe_load(text)
    input_root = args.input_root.resolve() if args.input_root else run / "input"
    run.mkdir(parents=True)
    if args.input_root is None:
        placeholder_inputs(config, input_root)
    # The example as written, its two root lines replaced: its comments stay.
    roots = {"root_input_path": input_root, "root_output_path": run / "output"}
    lines = [f"{key}: {json.dumps(str(roots[key]))}" if (key := line.split(":", 1)[0]) in roots else line
             for line in text.splitlines()]
    (run / "config.yaml").write_text("\n".join(lines) + "\n")
    print(run / "config.yaml")


if __name__ == "__main__":
    main()
