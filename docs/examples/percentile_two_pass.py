"""Manual two-pass sample-level percentile normalization on three synthetic FOVs.

Pass 1 runs the steps before percentile normalization (a default white
top-hat) on each FOV, summarizes the histograms at the normalization step's
input, merges them and writes the supplied-statistics file. Pass 2 applies the
full recipe with fit="supplied". Snakemake wiring of the two passes is §2.13.

From src/python: uv run python ../../docs/examples/percentile_two_pass.py OUTPUT_DIR
"""
from pathlib import Path
import sys

import numpy as np

from starfinder.dataset import Dataset, PipelineConfig, RoundState
from starfinder.image import ImageMetadata
from starfinder.preprocessing import (PercentileNormalizationConfig, PreprocessingRecipe, PreprocessingStep, TophatConfig,
    merge_histograms, read_supplied_statistics, summarize_histograms, summary_stage, supplied_section,
    supplied_statistics, write_histograms, write_supplied_statistics)

ROUNDS = ("round1", "round2")
CHANNELS = ("ch00", "ch01")


def fov_rounds(seed, puncta):
    """uint8 rounds of (4, 32, 32, 2): background noise plus `puncta` bright voxels per channel."""
    rng = np.random.default_rng(seed)
    rounds = {}
    for name in ROUNDS:
        volume = rng.normal(20, 3, (4, 32, 32, 2))
        for c in range(2):
            z, y, x = (rng.integers(0, n, puncta) for n in (4, 32, 32))
            volume[z, y, x, c] += rng.uniform(80, 200, puncta) * (1 + c)
        rounds[name] = np.clip(np.rint(volume), 0, 255).astype(np.uint8)
    return rounds


def load(dataset, fov_id, rounds):
    """Resident FOV holding copies of the raw rounds."""
    fov = dataset.fov(fov_id)
    for name, volume in rounds.items():
        fov.images[name] = volume.copy()
        fov.metadata[name] = ImageMetadata(f"{fov_id}/{name}")
    return fov


def main(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dataset = Dataset(output, output / "out", "example", "sample", "out",
                      rounds=RoundState(sequencing_rounds=list(ROUNDS), reference_round="round1"),
                      channel_order=CHANNELS)
    # Dense, sparse and near-empty FOVs.
    raw = {"FOV_001": fov_rounds(1, 200), "FOV_002": fov_rounds(2, 20), "FOV_003": fov_rounds(3, 1)}
    supplied = output / "supplied.json"
    recipe = PreprocessingRecipe((PreprocessingStep(TophatConfig()),
                                  PreprocessingStep(PercentileNormalizationConfig(fit="supplied"))),
                                 supplied_statistics=supplied)

    # Pass 1: summarize at the normalization step's input, after the top-hat.
    prefix, after = summary_stage(recipe, "percentile_normalization")
    summaries = []
    for fov_id, rounds in raw.items():
        fov = load(dataset, fov_id, rounds).run(PipelineConfig(preprocessing=prefix))
        summary = summarize_histograms({name: fov.images[name] for name in ROUNDS}, channel_labels=CHANNELS,
                                       fov_id=fov_id, summarized_after=after)
        write_histograms(summary, output / f"{fov_id}_histograms.npz")
        summaries.append(summary)
    merged = merge_histograms(summaries)
    section = supplied_section(recipe.steps[1].config, merged)
    write_supplied_statistics(supplied_statistics(merged, {"percentile_normalization": section}), supplied)

    # Pass 2: apply the full recipe with the supplied sample-level range.
    statistics = read_supplied_statistics(supplied, recipe, channel_labels=CHANNELS, rounds=ROUNDS)
    results = {}
    for fov_id, rounds in raw.items():
        fov = load(dataset, fov_id, rounds).run(PipelineConfig(preprocessing=recipe))
        results[fov_id] = {name: fov.images[name] for name in ROUNDS}
        assert all(image.dtype == np.uint8 for image in results[fov_id].values())
        assert fov.preprocessing_record["rounds"]["round1"][1]["fitted"] == section["fitted"]["round1"]
    for name, fitted in statistics["steps"]["percentile_normalization"]["fitted"].items():
        print(f"{name}: low {fitted['low']}, high {fitted['high']} from FOVs {statistics['fovs_used']}")
    return statistics, results


if __name__ == "__main__":
    main(sys.argv[1])
