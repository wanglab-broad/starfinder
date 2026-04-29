#!/usr/bin/env python3
"""Generate all-round montage examples for barcode decoding outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qc_codebook_aware_rescues import (
    DEFAULT_RESULT_DIR,
    StackCache,
    _json_safe,
    annotate_example,
    codebook_args,
    load_tensor,
    read_decoded,
    read_spots,
    save_all_round_montage,
    stack_dir_for,
)

from starfinder.barcode import load_codebook
from starfinder.barcode.codebook_aware import channel_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--dataset", default="aging")
    parser.add_argument("--fov", default="Position400")
    parser.add_argument("--config", default="balanced")
    parser.add_argument("--examples-per-category", type=int, default=6)
    parser.add_argument("--patch-radius", type=int, default=18)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--codebook-path", type=Path, default=None)
    parser.add_argument("--split-index", type=int, default=None)
    return parser.parse_args()


def h1_corrected_round_metrics(decoded: pd.DataFrame, tensor: np.ndarray) -> pd.DataFrame:
    h1 = decoded.loc[
        decoded["call_type"].fillna("").astype(str).str.startswith("rescued_h")
    ]
    rows = []
    for row in h1.itertuples(index=False):
        corrected_rounds = str(getattr(row, "corrected_rounds", "")).split(",")
        if not corrected_rounds or corrected_rounds[0] == "":
            continue
        round_idx = int(float(corrected_rounds[0]))
        wta_seq = str(getattr(row, "color_seq_wta"))
        decoded_seq = str(getattr(row, "decoded_seq"))
        if round_idx >= len(wta_seq) or round_idx >= len(decoded_seq):
            continue
        wta_color = wta_seq[round_idx]
        decoded_color = decoded_seq[round_idx]
        if not wta_color.isdigit() or not decoded_color.isdigit():
            continue

        spot_id = int(getattr(row, "spot_id"))
        values = np.asarray(tensor[spot_id : spot_id + 1], dtype=np.float64)
        probs = channel_probabilities(values)[0]
        intensities = values[0]

        wta_ch = int(wta_color) - 1
        target_ch = int(decoded_color) - 1
        wta_intensity = float(intensities[wta_ch, round_idx])
        target_intensity = float(intensities[target_ch, round_idx])
        wta_prob = float(probs[wta_ch, round_idx])
        target_prob = float(probs[target_ch, round_idx])
        rows.append(
            {
                "spot_id": spot_id,
                "corrected_round_0based": round_idx,
                "wta_color": wta_color,
                "target_color": decoded_color,
                "wta_intensity": wta_intensity,
                "target_intensity": target_intensity,
                "target_wta_intensity_ratio": (
                    target_intensity / wta_intensity if wta_intensity > 0 else np.nan
                ),
                "wta_prob": wta_prob,
                "target_prob": target_prob,
                "target_wta_prob_ratio": target_prob / wta_prob if wta_prob > 0 else np.nan,
                "target_minus_wta_prob": target_prob - wta_prob,
            }
        )
    return pd.DataFrame(rows)


def diversity_head(
    frame: pd.DataFrame,
    *,
    n: int,
    gene_column: str = "gene_base",
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    selected_indices = []
    seen_genes = set()
    for idx, row in frame.iterrows():
        gene = row.get(gene_column)
        if gene not in seen_genes:
            selected_indices.append(idx)
            seen_genes.add(gene)
        if len(selected_indices) >= n:
            break

    if len(selected_indices) < n:
        for idx in frame.index:
            if idx in selected_indices:
                continue
            selected_indices.append(idx)
            if len(selected_indices) >= n:
                break

    return frame.loc[selected_indices].copy()


def with_category(frame: pd.DataFrame, category: str) -> pd.DataFrame:
    result = frame.copy()
    result["category"] = category
    return result


def select_decoding_examples(
    merged: pd.DataFrame,
    h1_metrics: pd.DataFrame,
    *,
    n: int,
    seed: int,
) -> pd.DataFrame:
    del seed
    exact = merged.loc[merged["call_type"] == "exact"].copy()
    exact_high = exact.loc[
        (exact["geomean_prob"] >= 0.90)
        & (exact["min_round_margin"] >= 0.50)
        & (exact["mean_total_intensity"] >= exact["mean_total_intensity"].quantile(0.75))
    ]
    if len(exact_high) < n:
        exact_high = exact
    exact_high = exact_high.sort_values(
        ["geomean_prob", "min_round_margin", "mean_total_intensity"],
        ascending=[False, False, False],
    )
    high_wta = diversity_head(exact_high, n=n)

    h1 = merged.loc[
        merged["call_type"].fillna("").astype(str).str.startswith("rescued_h")
    ].merge(h1_metrics, on="spot_id", how="left")
    h1["score_delta_sort"] = h1["score_delta"].replace([np.inf, -np.inf], np.nan)
    h1["score_delta_sort"] = h1["score_delta_sort"].fillna(h1["score_delta_sort"].max())

    high_rescued = h1.loc[
        (h1["geomean_prob"] >= 0.68)
        & (h1["target_wta_intensity_ratio"] >= 0.90)
        & (h1["corrected_round_margin"] <= 0.15)
        & (h1["mean_total_intensity"] >= h1["mean_total_intensity"].quantile(0.50))
    ].sort_values(
        [
            "geomean_prob",
            "target_wta_intensity_ratio",
            "score_delta_sort",
            "mean_total_intensity",
        ],
        ascending=[False, False, False, False],
    )
    if len(high_rescued) < n:
        high_rescued = h1.loc[
            (h1["geomean_prob"] >= 0.68)
            & (h1["target_wta_intensity_ratio"] >= 0.90)
            & (h1["corrected_round_margin"] <= 0.15)
        ].sort_values(
            [
                "geomean_prob",
                "target_wta_intensity_ratio",
                "score_delta_sort",
                "mean_total_intensity",
            ],
            ascending=[False, False, False, False],
        )
    high_rescued = diversity_head(high_rescued, n=n)

    low_rescued = h1.loc[
        (h1["score_delta"] <= 0.40)
        | (h1["geomean_prob"] <= 0.53)
        | (h1["target_wta_intensity_ratio"] <= 0.65)
    ].sort_values(
        ["score_delta_sort", "geomean_prob", "target_wta_intensity_ratio"],
        ascending=[True, True, True],
    )
    low_rescued = diversity_head(low_rescued, n=n)

    no_call = merged.loc[merged["call_type"] == "no_call"].copy()
    no_call = no_call.loc[
        no_call["reject_reason"].isin(
            ["no_candidate", "geomean_prob_too_low", "ambiguous_candidate"]
        )
    ]
    no_call = no_call.sort_values(
        ["min_round_margin", "mean_total_intensity"],
        ascending=[True, True],
    )
    cannot_rescue = no_call.head(n).copy()

    selected = pd.concat(
        [
            with_category(high_wta, "01_high_confidence_WTA_exact"),
            with_category(high_rescued, "02_high_confidence_rescued_h1"),
            with_category(low_rescued, "03_low_confidence_rescued_h1"),
            with_category(cannot_rescue, "04_cannot_rescue_potential_noise"),
        ],
        ignore_index=True,
    )
    selected.insert(0, "example_id", np.arange(len(selected), dtype=int))
    return selected


def summarize_selection(examples: pd.DataFrame) -> pd.DataFrame:
    if examples.empty:
        return pd.DataFrame()
    aggregations: dict[str, Any] = {
        "n": ("spot_id", "size"),
        "median_geomean_prob": ("geomean_prob", "median"),
        "median_min_round_margin": ("min_round_margin", "median"),
        "median_mean_total_intensity": ("mean_total_intensity", "median"),
    }
    if "target_wta_intensity_ratio" in examples:
        aggregations.update(
            {
                "median_target_wta_intensity_ratio": (
                    "target_wta_intensity_ratio",
                    "median",
                ),
                "median_target_prob": ("target_prob", "median"),
                "median_wta_prob": ("wta_prob", "median"),
            }
        )
    return examples.groupby("category", dropna=False).agg(**aggregations).reset_index()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        args.result_dir
        / args.dataset
        / "qc"
        / f"{args.fov}_{args.config}_decoding_examples"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    decoded = read_decoded(args.result_dir, args.dataset, args.config, args.fov)
    spots = read_spots(args.result_dir, args.dataset, args.fov)
    merged = decoded.merge(spots, on="spot_id", how="left")
    tensor = load_tensor(args.result_dir, args.dataset, args.fov)
    n_channels, n_rounds = int(tensor.shape[1]), int(tensor.shape[2])

    codebook_path, split_index = codebook_args(
        args.dataset,
        args.fov,
        args.codebook_path,
        args.split_index,
        result_dir=args.result_dir,
    )
    _gene_to_seq, seq_to_gene = load_codebook(codebook_path, split_index=split_index)

    h1_metrics = h1_corrected_round_metrics(merged, tensor)
    selected = select_decoding_examples(
        merged,
        h1_metrics,
        n=args.examples_per_category,
        seed=args.seed,
    )
    annotated = pd.DataFrame(
        [
            annotate_example(row, tensor=tensor, seq_to_gene=seq_to_gene)
            for _, row in selected.iterrows()
        ]
    )

    stack_cache = StackCache(
        stack_dir=stack_dir_for(args.dataset, args.fov, args.result_dir),
        handles={},
    )
    montage_paths = []
    try:
        for _, row in annotated.iterrows():
            path = save_all_round_montage(
                row,
                stack_cache=stack_cache,
                output_dir=output_dir,
                patch_radius=args.patch_radius,
                n_rounds=n_rounds,
                n_channels=n_channels,
            )
            montage_paths.append(str(path))
    finally:
        stack_cache.close()

    annotated["all_round_montage_path"] = montage_paths
    annotated.to_csv(output_dir / "decoding_examples.csv", index=False)
    summary = summarize_selection(annotated)
    summary.to_csv(output_dir / "decoding_example_summary.csv", index=False)

    run_summary = {
        "dataset": args.dataset,
        "fov": args.fov,
        "config": args.config,
        "n_examples": int(len(annotated)),
        "examples_per_category": args.examples_per_category,
        "output_dir": str(output_dir),
        "codebook_path": str(codebook_path),
        "split_index": split_index,
        "category_counts": annotated["category"].value_counts().sort_index().to_dict(),
    }
    (output_dir / "decoding_example_summary.json").write_text(
        json.dumps(run_summary, indent=2, default=_json_safe)
    )

    print(
        f"Saved {len(annotated)} decoding example all-round montages under "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
