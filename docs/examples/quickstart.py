"""Generate tiny synthetic TIFFs and process them to molecule-level CSVs.

Run from src/python with a new external output directory; see docs/recipes.md, "Quick start".
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from starfinder.assignment import load_molecules
from starfinder.dataset import CheckpointConfig, Dataset, PipelineConfig, RoundState
from starfinder.io import ImageLoadConfig
from starfinder.readout import NeighborhoodSumConfig, ReadFilterConfig, WtaDecoderConfig
from starfinder.registration import RegistrationRecipe, RegistrationStep, TranslationConfig
from starfinder.spot_finding import LocalMaximaConfig
from starfinder.synthetic import (BENCHMARK_PRESETS, benchmark_scene_preset, generate_amplicon_dataset,
                                  save_amplicon_dataset)


def main(output: Path) -> None:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # The tiny benchmark preset at seed 42, the data of `starfinder synthetic generate --mode e2e --preset tiny`.
    codebook, config = benchmark_scene_preset("tiny")
    fov_ids = tuple(f"FOV_{i + 1:03d}" for i in range(BENCHMARK_PRESETS["tiny"]["fovs"]))
    generated = generate_amplicon_dataset(codebook, replace(config, seed=42), fov_ids=fov_ids, preset="tiny")
    # Images in the round/FOV layout the dataset loader reads, the codebook and the truth files.
    save_amplicon_dataset(generated, output / "input")
    truth = generated.historical_truth
    rounds = list(generated.codebook.round_labels)
    channels = list(generated.channel_labels)
    del generated  # The images are on disk; each FOV loads its own below.

    dataset = Dataset(
        input_root=output / "input",
        output_root=output / "results",
        dataset_id="tiny",
        sample_id="synthetic",
        output_id="quickstart",
        rounds=RoundState(sequencing_rounds=rounds, reference_round="round1"),
        channel_order=channels,
        fov_pattern="FOV_{i:03d}",
    )
    dataset.rounds.validate()
    # The {i} pattern names the FOVs from 1, as the workflow numbers them. fovs.csv, the FOV membership the Python
    # workflow reads from the input root, lists them as one sample.
    fov_ids = dataset.fov_ids(len(truth["fovs"]), start=1)
    pd.DataFrame({"sample": "synthetic", "fov_id": fov_ids}).to_csv(output / "input" / "fovs.csv", index=False)
    dataset.load_codebook(output / "input" / "codebook.csv")
    # Load, register, detect, extract, decode and filter in one run.
    pipeline = PipelineConfig(
        load=ImageLoadConfig(channel_labels=tuple(channels)),
        registration=RegistrationRecipe((RegistrationStep(TranslationConfig()),)),
        spot_finding=LocalMaximaConfig(threshold_mode="noise", threshold_value=5.0, min_distance_voxels=1),
        extraction=NeighborhoodSumConfig((1, 2, 2)),
        decoding=WtaDecoderConfig(diagnostics=True),
        filtering=ReadFilterConfig(),
    )
    summary = {"preset": "tiny", "seed": truth["seed"], "fovs": {}}
    for fov_id in fov_ids:
        fov = dataset.fov(fov_id)
        # stages=() writes the run record fovs/<fov>/run.json and no stage files.
        fov.run(pipeline, checkpoints=CheckpointConfig(stages=()))
        for volume in fov.images.values():
            assert volume.shape == (8, 128, 128, 4)
            assert volume.dtype == np.uint16

        # These are completion checks, not a benchmark of detection accuracy.
        assert 0 < len(fov.filtering_result.accepted) <= len(fov.spot_result.spots)
        assert set(fov.filtering_result.accepted["gene_id"]) <= set(dataset.codebook.genes)
        assert fov.filtering_result.accepted["observed_color_sequence"].str.fullmatch(r"[1-4]{4}").all()
        detected_shifts = {}
        for round_name, shift in ((name, results[0].transform.displacement_zyx) for name, results in fov.registration_results.items()):
            expected = truth["fovs"][fov_id]["shifts"][round_name]
            np.testing.assert_allclose(shift, expected, atol=1, rtol=0)
            detected_shifts[round_name] = [float(value) for value in shift]

        # The reads table, on request, keeps diagnostic columns; molecules.csv holds the accepted molecules.
        reads_path = fov.export_spots("reads", columns=["spot_namespace", "spot_id", "z", "y", "x", "gene_id",
                                                        "observed_color_sequence", "call_status"])
        molecules_path = fov.export_spots()
        # The run record holds the round roles, the registration attempts and transforms, the counts, the
        # filter's fractions and the read summary; export_spots added the table as outputs.molecules.
        run_record = json.loads(fov.paths.run_record.read_text())
        assert run_record["status"] == "succeeded" and run_record["rounds"]["reference_round"] == "round1"
        assert run_record["counts"]["filtering"]["accepted"] == len(fov.filtering_result.accepted)
        assert run_record["counts"]["filtering"]["fractions"]["accepted"] == fov.filtering_result.fractions["accepted"]
        assert run_record["outputs"]["molecules"]["rows"] == len(fov.filtering_result.accepted)
        assert {name: entry["results"][0]["transform"]["displacement_zyx"]
                for name, entry in run_record["transforms"].items()} == detected_shifts
        reads_saved = pd.read_csv(reads_path, dtype={"observed_color_sequence": str}, na_values=["<NA>"],
                                  keep_default_na=False)
        molecules = load_molecules(molecules_path, genes=dataset.codebook.genes)
        assert list(pd.read_csv(molecules_path, nrows=0).columns) == [
            "spot_namespace", "spot_id", "z", "y", "x", "gene_id", "entry_id", "qc_score"]
        assert len(reads_saved) == len(fov.spot_result.spots)
        assert len(molecules) == len(fov.filtering_result.accepted)
        # Zero-based coordinates, equal to the detection's, with the spot identities kept.
        accepted = fov.spot_result.spots.merge(fov.filtering_result.accepted[["spot_id"]], on="spot_id",
                                               validate="one_to_one")
        assert molecules.table.spot_id.tolist() == accepted.spot_id.tolist()
        np.testing.assert_array_equal(molecules.table[["z", "y", "x"]], accepted[["z", "y", "x"]])
        for axis, size in zip(("z", "y", "x"), (8, 128, 128)):
            assert reads_saved[axis].between(0, size - 1).all()
        record = {
            "generated_spots": len(truth["fovs"][fov_id]["spots"]),
            "detected_spots": len(fov.spot_result.spots),
            "retained_reads": len(fov.filtering_result.accepted),
            "detected_shifts_zyx": detected_shifts,
            "gene_counts": {gene: int(n) for gene, n in molecules.table["gene_id"].value_counts().items()},
            "molecules_csv": str(molecules_path.relative_to(output)),
        }
        summary["fovs"][fov_id] = record
        print(f"{fov_id}: {record['detected_spots']} detected, {record['retained_reads']} retained")
        print(molecules.table.head().to_string(index=False))
        del fov  # Process FOVs sequentially without retaining their image arrays.

    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Quickstart checks passed. Outputs: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New output directory outside the checkout")
    main(parser.parse_args().output)
