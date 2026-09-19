"""Bounded maintained-script imports and saved-data adapters using public diagnostics."""

import importlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from .barcode_cases import tensor


def test_four_scripts_public_diagnostics_and_saved_tensor_smoke(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    names = (
        "diagnose_codebook_aware_rescues",
        "generate_decoding_example_montages",
        "qc_codebook_aware_rescues",
        "run_codebook_aware_benchmark",
    )
    modules = {name: importlib.import_module(name) for name in names}
    adapter = importlib.import_module("_decoding_inputs")
    values = tensor(["4422", "4322", "1111"])
    result = adapter.saved_decoding(values, {"4422": "A"}, spot_ids=["8", "3", "5"])
    assert result.table.spot_id.tolist() == ["8", "3", "5"]
    probs = modules["qc_codebook_aware_rescues"].probs_for_spot(values, 1)
    np.testing.assert_allclose(probs.sum(axis=0), 1)
    assert (
        modules["qc_codebook_aware_rescues"].best_target_candidate(probs, ["4422", "4222"])
        == "4222"
    )
    decoded = pd.DataFrame({"call_group": ["no_call"], "color_seq_wta": ["4322"]})
    candidates = modules["qc_codebook_aware_rescues"].target_no_call_candidates(
        decoded, {"4422": "A"}, "A", 4, 4
    )
    assert candidates.tolist() == ["4422"]
    spots = pd.DataFrame(
        {"spot_id": [8, 3, 5], "z": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0]}
    )
    report = adapter.report_table(result)
    evaluated = modules["run_codebook_aware_benchmark"].decoded_eval_frame(
        spots, report, use_wta=False
    )
    assert evaluated.gene.iloc[0] == "A" and len(evaluated) == 3
    # Exercise the retained report writer on three supplied tensor rows, no image pipeline.
    rows = modules["run_codebook_aware_benchmark"].run_dataset_fov(
        dataset="contract",
        fov_id="fov",
        tensor=values,
        spots=spots,
        seq_to_gene={"4422": "A"},
        output_dir=tmp_path,
        configs=["wta_exact"],
    )
    assert len(rows) == 1
    assert (tmp_path / "contract/codebook_aware/wta_exact/fov.csv").exists()


def test_script_extraction_existing_small_fixture(monkeypatch, small_dataset):
    from starfinder.io import load_round, ImageLoadConfig

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    adapter = importlib.import_module("_decoding_inputs")
    images = {}
    for r in ("round1", "round2", "round3", "round4"):
        images[r] = load_round(
            small_dataset / "FOV_001" / r,
            config=ImageLoadConfig(channel_labels=("ch00", "ch01", "ch02", "ch03")),
        ).image
    spots = pd.DataFrame({"spot_id": ["original"], "z": [5.0], "y": [16.0], "x": [16.0]})
    values = adapter.saved_extraction(
        images, spots, list(images), (0, 0, 0), namespace="small/FOV_001"
    )
    assert values.shape == (1, 4, 4)
    for i, image in enumerate(images.values()):
        np.testing.assert_array_equal(values[0, :, i], image[5, 16, 16])
    book = adapter.saved_codebook(small_dataset / "codebook.csv")
    result = adapter.saved_decoding(values, book.seq_to_gene, spot_ids=spots.spot_id)
    assert result.table.spot_id.tolist() == ["original"]
