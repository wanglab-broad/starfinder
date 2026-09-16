"""Materialize a schema-validated, dry-run-only workflow example outside Git.

No microscopy images are generated. Never execute this placeholder input tree.
Run from src/python with uv run --with 'snakemake>=9,<10' python ...
"""

import argparse
from pathlib import Path

import yaml
from snakemake.utils import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory outside the checkout")
    parser.add_argument("--template", choices=["minimal", "full"], default="minimal")
    parser.add_argument("--mode", choices=["free", "direct", "subtile", "deep"])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output == repo or repo in output.parents:
        parser.error("output must be outside the checkout")
    if output.exists():
        parser.error("output must be a new directory (existing runs are preserved)")
    source = Path(__file__).with_name(f"workflow-{args.template}.yaml")
    config = yaml.safe_load(source.read_text())
    config.update(
        config_path=str(output / "config.yaml"),
        starfinder_path=str(repo),
        root_input_path=str(output / "input"),
        root_output_path=str(output / "output"),
        envs_path=str(output / "unavailable-envs"),
        fiji_path=str(output / "unavailable-fiji"),
    )
    if args.mode:
        config["workflow_mode"] = args.mode
    required = {
        "subtile": {"gr_single_fov_subtile", "lrsf_single_fov_subtile", "stitch_subtile"},
        "deep": {"deep_create_subtile", "deep_rsf_subtile", "stitch_subtile"},
    }.get(config["workflow_mode"], {"rsf_single_fov"})
    if not required.issubset(config["rules"]):
        parser.error("subtile/deep require --template full")
    validate(config, str(repo / "workflow/schemas/config.schema.yaml"))
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    input_dir = output / "input" / config["dataset_id"] / config["sample_id"]
    input_dir.mkdir(parents=True)
    (input_dir / "genes.csv").write_text("ExampleGene,ACGTCC\n")
    for r in range(1, config["n_rounds"] + 1):
        (input_dir / f"round{r}" / "tile_1").mkdir(parents=True)
    doc_dir = output / "output" / config["dataset_id"] / config["output_id"] / "documents"
    doc_dir.mkdir(parents=True)
    (doc_dir / "sample-annotation.csv").write_text("sample_id,fov_start,fov_end\nsample,1,1\n")
    (output / "DRY_RUN_ONLY.txt").write_text(
        "Empty round/FOV directories satisfy DAG existence checks only.\n"
        "No TIFF, MATLAB, segmentation, assignment or scientific validation.\n"
    )
    print(f"Validated {source.name}; prepared {output / 'config.yaml'} for --dry-run only")


if __name__ == "__main__":
    main()
