"""Write the mid-size synthetic sample of the workflow's local rehearsal and its configuration (W-761).

The rehearsal runs the whole workflow at size under both engines (Snakemake and Nextflow) on this scene before the
runs on Broad UGER (design/validation/orchestration.md, U4). The scene is ``tiny_sample``'s
(``scripts/workflow_golden.py fixture``: ``sample_scene_preset("sample-tiny-v1")``, seed 42, four rounds of four
channels) grown to four overlapping FOVs of 30 × 1024 × 1024 in a 2 × 2 grid, column by column, with an overlap of
0.1; its cells are grown with it (``CELLS``). The configuration is the full example's
(``docs/examples/workflow-full.yaml``) on that scene, so that only the size differs from the Quick start of the
workflow guide. Only the package's public API is used. At the default size the scene took 32 min and 8.4 GB of peak
RSS on one CPU (W-761), and its inputs take 8.8 GB.

Subcommands, each run in its own process:

``scene WORK [--shape Z Y X] [--nuclei N]``
    Generate the scene and write the workflow inputs under ``WORK/input/<DATASET_ID>/sample/``, in the layout of
    ``workflow_golden.py fixture``: one float32 OME-TIFF per round, FOV and channel, the nuclear stain as the
    reference round's ``ch04.tif``, an additional round ``flamingo`` (``ch04.tif`` the stain, ``ch05.tif`` the
    reference round's channel maximum), these three as uint16, ``genes.csv``, ``fovs.csv``,
    ``positions.csv`` and the truth cell labels as ``labels/<fov>.tif``, which both imported segmentation runs read.
    Each FOV is written as it completes (``generate_sample_scene``'s ``on_fov``) and not kept in memory.
    ``WORK/scene.json`` lists every file with its bytes and SHA-256, and the scene's settings. ``--shape`` and
    ``--nuclei`` make a smaller scene of the same kind (the test plans one of 8 × 64 × 64).
``config WORK OUTPUT_ROOT``
    Write ``WORK/config.yaml``: the full example with the two roots set to ``WORK/input`` and ``OUTPUT_ROOT``,
    the dataset and output ids of this scene, and every key the example writes otherwise unchanged
    (``execution.tasks.fov.threads`` 1, as there). Print its path.

Run from src/python with the package's environment, for example:

    uv run python scripts/workflow_rehearsal_scene.py scene WORK
    uv run python scripts/workflow_rehearsal_scene.py config WORK OUTPUT_ROOT
    uv run starfinder workflow plan --config WORK/config.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EXAMPLE = REPO / "docs" / "examples" / "workflow-full.yaml"
BASE_PRESET = "sample-tiny-v1"
DATASET_ID = "workflow-midsize-v1"
SAMPLE_ID = "sample"
OUTPUT_ID = "rehearsal"
SHAPE_ZYX = (30, 1024, 1024)
GRID = dict(columns=2, rows=2, order="column_major_down_right", overlap=0.1)
# The cells, grown with the FOVs: nuclei of about 5 µm lateral and 3 µm axial radius at the preset's spacing
# (1.0, 0.5, 0.5) µm, territories reaching about 6 µm beyond them, and 400 seeds over the 2 × 2 grid's domain of about
# 973 × 973 × 30 µm. Values for an engineering rehearsal, not a tissue model.
CELLS = dict(nuclei=400, binucleate=8, empty=8, nucleus_axial_radius=3.0, nucleus_lateral_radius=5.0, reach=6.0,
             molecules_per_cell=40.0, ambient_density=0.0005, max_count=200_000)
ROUNDS = ("round1", "round2", "round3", "round4")
CHANNELS = ("ch00", "ch01", "ch02", "ch03")
STAIN = "ch04"
FLAMINGO = "ch05"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def input_root(work):
    return Path(work) / "input" / DATASET_ID / SAMPLE_ID


def _uint16(image):
    """A float image rounded and clipped to uint16."""
    return np.ascontiguousarray(np.clip(np.rint(image), 0, np.iinfo(np.uint16).max).astype(np.uint16))


def scene_config(shape_zyx=SHAPE_ZYX, nuclei=CELLS["nuclei"]):
    """(codebook, SampleSceneConfig): ``sample-tiny-v1`` with this scene's grid, FOV shape and cells."""
    from starfinder.synthetic import AcquisitionConfig, CellsConfig, ScalarDistribution, sample_scene_preset
    book, config = sample_scene_preset(BASE_PRESET)
    shape = tuple(int(v) for v in shape_zyx)
    acquisition = AcquisitionConfig(columns=GRID["columns"], rows=GRID["rows"], order=GRID["order"],
                                    fov_shape_zyx=shape, spacing_zyx=config.acquisition.spacing_zyx,
                                    overlap=GRID["overlap"], jitter_max_zyx=config.acquisition.jitter_max_zyx)
    # float32 accumulation: the preset's float64 is for its oracle identities, not needed at this size.
    template = dataclasses.replace(config.fov, shape_zyx=shape, accumulation="float32", dataset_version=DATASET_ID,
                                   scene_key=DATASET_ID)
    def lognormal(median, sd):
        return ScalarDistribution("lognormal", (math.log(median), sd))

    # The preset's spreads: 0.1 for the nucleus radii, 0.15 for the reach.
    cells = CellsConfig(nuclei=int(nuclei), binucleate=min(CELLS["binucleate"], int(nuclei) // 4),
                        empty=min(CELLS["empty"], int(nuclei) // 4),
                        nucleus_axial_radius=lognormal(CELLS["nucleus_axial_radius"], 0.1),
                        nucleus_lateral_radius=lognormal(CELLS["nucleus_lateral_radius"], 0.1),
                        reach=lognormal(CELLS["reach"], 0.15))
    return book, dataclasses.replace(config, fov=template, acquisition=acquisition, cells=cells,
                                     molecules_per_cell=CELLS["molecules_per_cell"],
                                     ambient_density=CELLS["ambient_density"], max_count=CELLS["max_count"])


def write_scene(work, shape_zyx=SHAPE_ZYX, nuclei=CELLS["nuclei"]):
    """The scene's workflow inputs under ``work`` (a new or empty folder); return the record of scene.json."""
    import tifffile
    from starfinder.io import save_volume
    from starfinder.readout import colors_to_bases, load_codebook
    from starfinder.synthetic import generate_sample_scene
    work = Path(work)
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"{work} is not empty")
    book, config = scene_config(shape_zyx, nuclei)
    if tuple(book.round_labels) != ROUNDS or tuple(book.channel_labels) != CHANNELS:
        raise ValueError(f"unexpected preset labels {book.round_labels} {book.channel_labels}")
    root = input_root(work)
    root.mkdir(parents=True)

    def on_fov(fov, formed, stain, labels):
        metadata = formed.metadata
        for r in ROUNDS:
            for c, channel in enumerate(CHANNELS):
                # float32, as workflow_golden.py writes them: rounded to uint16, the noise's negative half is
                # clipped to 0, and the full example's spot finding then measures a noise MAD of 0 (W-761).
                save_volume(formed.rounds[r][..., c:c + 1], root / r / fov / f"{channel}.ome.tif", metadata=metadata)
        # Plain uint16 ZYX TIFFs, as workflow_golden.py writes the stain and the Flamingo channel.
        for folder, name, image in ((ROUNDS[0], STAIN, _uint16(stain)), ("flamingo", STAIN, _uint16(stain)),
                                    ("flamingo", FLAMINGO, _uint16(formed.rounds[ROUNDS[0]].max(axis=-1)))):
            (root / folder / fov).mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(root / folder / fov / f"{name}.tif", image, photometric="minisblack")
        # The truth cells as a legacy label file: a plain ZYX TIFF without stored metadata.
        (root / "labels").mkdir(parents=True, exist_ok=True)
        cells = labels["cell"].reshape(formed.rounds[ROUNDS[0]].shape[:3])
        tifffile.imwrite(root / "labels" / f"{fov}.tif", np.ascontiguousarray(cells), compression="zlib",
                         photometric="minisblack")

    scene = generate_sample_scene(book, config=config, on_fov=on_fov)
    fov_ids = list(scene.positions.fov_id)
    # genes.csv: headerless gene,barcode rows of the two_base encoding the workflow loads by default.
    book_rows = book.table.drop_duplicates("gene_id")
    lines = [f"{gene},{colors_to_bases(colors, 'C')[::-1]}"
             for gene, colors in zip(book_rows.gene_id, book_rows.color_sequence)]
    (root / "genes.csv").write_text("\n".join(lines) + "\n")
    loaded = load_codebook(root / "genes.csv", round_labels=ROUNDS, channel_labels=CHANNELS)
    if dict(loaded.gene_to_seq) != dict(book.gene_to_seq):
        raise ValueError("genes.csv does not load as the preset's codebook")
    (root / "fovs.csv").write_text("sample,fov_id\n" + "".join(f"{SAMPLE_ID},{fov}\n" for fov in fov_ids))
    scene.positions.to_csv(root / "positions.csv", index=False)
    files = {p.relative_to(work).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in sorted(work.rglob("*")) if p.is_file()}
    record = {"base_preset": BASE_PRESET, "dataset_id": DATASET_ID, "seed": config.fov.seed, "fov_ids": fov_ids,
              "shape_zyx": list(config.fov.shape_zyx), "spacing_zyx": list(config.acquisition.spacing_zyx),
              "grid": GRID, "cells": dict(CELLS, nuclei=config.cells.nuclei, binucleate=config.cells.binucleate,
                                          empty=config.cells.empty),
              "counts": {"nuclei": int(len(scene.nuclei)), "cells": int(len(scene.cells)),
                         "molecules": int(len(scene.molecules))},
              "rounds": list(ROUNDS), "channels": list(CHANNELS), "files": files}
    (work / "scene.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def write_config(work, output_root):
    """WORK/config.yaml: the full example on this scene, its two roots and the scene's ids set."""
    import yaml
    work = Path(work).resolve()
    config = yaml.safe_load(EXAMPLE.read_text())
    config.update(root_input_path=str(work / "input"), root_output_path=str(Path(output_root).resolve()),
                  dataset_id=DATASET_ID, sample_id=SAMPLE_ID, output_id=OUTPUT_ID)
    path = work / "config.yaml"
    path.write_text(f"# The full example ({EXAMPLE.relative_to(REPO)}) on {DATASET_ID}, written by "
                    f"{Path(__file__).name}.\n" + yaml.safe_dump(config, sort_keys=False))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    scene = commands.add_parser("scene", help="generate the scene and write the workflow inputs")
    scene.add_argument("work", type=Path, help="a new or empty folder")
    scene.add_argument("--shape", type=int, nargs=3, default=SHAPE_ZYX, metavar=("Z", "Y", "X"),
                       help=f"the FOV shape (default {' '.join(map(str, SHAPE_ZYX))})")
    scene.add_argument("--nuclei", type=int, default=CELLS["nuclei"], help="the ordinary cell seeds")
    config = commands.add_parser("config", help="write WORK/config.yaml")
    config.add_argument("work", type=Path, help="the folder of the scene")
    config.add_argument("output_root", type=Path, help="the configuration's root_output_path")
    args = parser.parse_args(argv)
    if args.command == "scene":
        record = write_scene(args.work, tuple(args.shape), args.nuclei)
        print(json.dumps({k: record[k] for k in ("fov_ids", "shape_zyx", "counts")}))
    else:
        print(write_config(args.work, args.output_root))


if __name__ == "__main__":
    main()
