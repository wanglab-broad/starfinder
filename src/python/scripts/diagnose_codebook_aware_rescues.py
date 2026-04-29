#!/usr/bin/env python3
"""Diagnose whether codebook-aware rescued reads look artifact-like."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BENCHMARK_ROOT = Path("/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark")
DEFAULT_RESULT_DIR = BENCHMARK_ROOT / "decoding" / "codebook_aware" / "results"
PAD_SUFFIX_RE = re.compile(r"_(STAR|RIBO)_pad_\d+$")
RAW_REAL_CODEBOOKS: dict[str, tuple[Path, int | None]] = {
    "cell_culture_3D": (
        Path("/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/cell-culture-3D/genes.csv"),
        None,
    ),
    "tissue_2D": (
        Path("/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/tissue-2D/genes.csv"),
        None,
    ),
}

CONFIDENCE_METRICS = [
    "score",
    "score_delta",
    "geomean_prob",
    "min_round_margin",
    "corrected_round_margin",
    "mean_total_intensity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--dataset", default="aging")
    parser.add_argument("--config", default="balanced")
    parser.add_argument("--baseline-config", default="wta_exact")
    parser.add_argument("--fovs", nargs="+", default=None)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--edge-margin-px", type=float, default=50.0)
    parser.add_argument("--min-bin-reads", type=int, default=1000)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--codebook-path", type=Path, default=None)
    parser.add_argument("--split-index", type=int, default=None)
    return parser.parse_args()


def strip_pad_suffix(gene_name: Any) -> str | None:
    if gene_name is None or pd.isna(gene_name):
        return None
    value = str(gene_name)
    if value == "":
        return None
    return PAD_SUFFIX_RE.sub("", value)


def is_assigned(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str) != "")


def base_gene_counts(series: pd.Series) -> pd.Series:
    genes = series.loc[is_assigned(series)].map(strip_pad_suffix).dropna()
    return genes.value_counts()


def rank_series(counts: pd.Series) -> pd.Series:
    if counts.empty:
        return pd.Series(dtype="Int64")
    return counts.rank(method="min", ascending=False).astype("Int64")


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def bool_value(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return bool(value)


def read_inputs(
    result_dir: Path,
    dataset: str,
    config: str,
    baseline_config: str,
    fov_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decoded_path = result_dir / dataset / "codebook_aware" / config / f"{fov_id}.csv"
    baseline_path = (
        result_dir / dataset / "codebook_aware" / baseline_config / f"{fov_id}.csv"
    )
    spots_path = result_dir / dataset / "spots" / f"{fov_id}.csv"
    missing = [
        path
        for path in [decoded_path, baseline_path, spots_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required diagnostic inputs: "
            + ", ".join(str(path) for path in missing)
        )

    decoded = pd.read_csv(decoded_path, low_memory=False)
    baseline = pd.read_csv(baseline_path, low_memory=False)
    spots = pd.read_csv(spots_path, low_memory=False)
    return decoded, baseline, spots


def normalized_call_group(call_type: pd.Series) -> pd.Series:
    values = call_type.fillna("").astype(str)
    return np.select(
        [
            values == "exact",
            values.str.startswith("rescued"),
            values == "no_call",
        ],
        ["exact", "rescued", "no_call"],
        default=values,
    )


def summarize_confidence(decoded: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = decoded.copy()
    work["call_group"] = normalized_call_group(work["call_type"])
    for call_group, group in work.groupby("call_group", dropna=False):
        for metric in CONFIDENCE_METRICS:
            if metric not in group:
                continue
            values_all = pd.to_numeric(group[metric], errors="coerce").dropna()
            finite_values = values_all[np.isfinite(values_all)]
            n_infinite = int(np.isinf(values_all).sum())
            if finite_values.empty:
                rows.append(
                    {
                        "call_group": call_group,
                        "metric": metric,
                        "n": int(values_all.size),
                        "n_finite": 0,
                        "n_infinite": n_infinite,
                    }
                )
                continue
            quantiles = finite_values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            rows.append(
                {
                    "call_group": call_group,
                    "metric": metric,
                    "n": int(values_all.size),
                    "n_finite": int(finite_values.size),
                    "n_infinite": n_infinite,
                    "mean": float(finite_values.mean()),
                    "std": float(finite_values.std(ddof=0)),
                    "min": float(finite_values.min()),
                    "q05": float(quantiles.loc[0.05]),
                    "q25": float(quantiles.loc[0.25]),
                    "median": float(quantiles.loc[0.5]),
                    "q75": float(quantiles.loc[0.75]),
                    "q95": float(quantiles.loc[0.95]),
                    "max": float(finite_values.max()),
                }
            )
    return pd.DataFrame(rows)


def make_gene_rescue_enrichment(
    decoded: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    rescue_mask = decoded["call_type"].fillna("").astype(str).str.startswith("rescued")
    exact_mask = decoded["call_type"] == "exact"

    exact_counts = base_gene_counts(decoded.loc[exact_mask, "gene"])
    rescued_counts = base_gene_counts(decoded.loc[rescue_mask, "gene"])
    final_counts = base_gene_counts(decoded["gene"])
    baseline_counts = base_gene_counts(baseline["gene"])
    final_ranks = rank_series(final_counts)
    baseline_ranks = rank_series(baseline_counts)

    genes = sorted(
        set(exact_counts.index)
        | set(rescued_counts.index)
        | set(final_counts.index)
        | set(baseline_counts.index)
    )
    rows = []
    total_rescued = int(rescue_mask.sum())
    for gene in genes:
        exact_count = int(exact_counts.get(gene, 0))
        rescued_count = int(rescued_counts.get(gene, 0))
        final_count = int(final_counts.get(gene, 0))
        baseline_count = int(baseline_counts.get(gene, 0))
        baseline_rank = baseline_ranks.get(gene, pd.NA)
        final_rank = final_ranks.get(gene, pd.NA)
        rows.append(
            {
                "gene_base": gene,
                "baseline_count": baseline_count,
                "exact_count": exact_count,
                "rescued_count": rescued_count,
                "final_count": final_count,
                "delta_vs_baseline": final_count - baseline_count,
                "rescue_fraction_of_gene_final": safe_ratio(
                    rescued_count,
                    final_count,
                ),
                "rescued_fraction_global": safe_ratio(rescued_count, total_rescued),
                "baseline_rank": baseline_rank,
                "final_rank": final_rank,
                "rank_delta": (
                    int(baseline_rank) - int(final_rank)
                    if not pd.isna(baseline_rank) and not pd.isna(final_rank)
                    else pd.NA
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["rescued_count", "final_count", "gene_base"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def make_top_shift(
    gene_table: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, float, list[str], list[str]]:
    baseline_top = set(
        gene_table.sort_values(["baseline_count", "gene_base"], ascending=[False, True])
        .head(top_n)["gene_base"]
        .tolist()
    )
    final_top = set(
        gene_table.sort_values(["final_count", "gene_base"], ascending=[False, True])
        .head(top_n)["gene_base"]
        .tolist()
    )
    union = baseline_top | final_top
    jaccard = safe_ratio(len(baseline_top & final_top), len(union))
    entered = sorted(final_top - baseline_top)
    exited = sorted(baseline_top - final_top)

    top_shift = gene_table.loc[gene_table["gene_base"].isin(union)].copy()
    top_shift["baseline_top"] = top_shift["gene_base"].isin(baseline_top)
    top_shift["final_top"] = top_shift["gene_base"].isin(final_top)
    top_shift["status"] = np.select(
        [
            top_shift["baseline_top"] & top_shift["final_top"],
            top_shift["final_top"] & ~top_shift["baseline_top"],
            top_shift["baseline_top"] & ~top_shift["final_top"],
        ],
        ["shared", "entered", "exited"],
        default="other",
    )
    top_shift = top_shift.sort_values(
        ["status", "final_count", "baseline_count", "gene_base"],
        ascending=[True, False, False, True],
    )
    return top_shift, jaccard, entered, exited


def top_jaccard_from_counts(
    baseline_counts: pd.Series,
    final_counts: pd.Series,
    top_n: int = 10,
) -> tuple[float, list[str], list[str]]:
    baseline_top = set(baseline_counts.sort_values(ascending=False).head(top_n).index)
    final_top = set(final_counts.sort_values(ascending=False).head(top_n).index)
    union = baseline_top | final_top
    jaccard = safe_ratio(len(baseline_top & final_top), len(union))
    return jaccard, sorted(final_top - baseline_top), sorted(baseline_top - final_top)


def make_gate_sensitivity(
    decoded: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    rescue_mask = decoded["call_type"].fillna("").astype(str).str.startswith("rescued")
    baseline_counts = base_gene_counts(baseline["gene"])
    score_delta = pd.to_numeric(decoded["score_delta"], errors="coerce")
    geomean_prob = pd.to_numeric(decoded["geomean_prob"], errors="coerce")
    corrected_margin = pd.to_numeric(decoded["corrected_round_margin"], errors="coerce")

    gates = [
        ("balanced_all", 0.25, 0.45, 0.20),
        ("score0.50_geo0.50_margin0.15", 0.50, 0.50, 0.15),
        ("score0.50_geo0.55_margin0.10", 0.50, 0.55, 0.10),
        ("score0.75_geo0.55_margin0.10", 0.75, 0.55, 0.10),
        ("score1.00_geo0.55_margin0.10", 1.00, 0.55, 0.10),
        ("score0.75_geo0.60_margin0.10", 0.75, 0.60, 0.10),
        ("score1.00_geo0.60_margin0.08", 1.00, 0.60, 0.08),
    ]

    rows = []
    for gate_name, min_score_delta, min_geomean_prob, max_corrected_margin in gates:
        keep_rescue = (
            rescue_mask
            & (score_delta >= min_score_delta)
            & (geomean_prob >= min_geomean_prob)
            & (corrected_margin.isna() | (corrected_margin <= max_corrected_margin))
        )
        posthoc_gene = decoded["gene"].where(~rescue_mask | keep_rescue)
        final_counts = base_gene_counts(posthoc_gene)
        rescued_counts = base_gene_counts(decoded.loc[keep_rescue, "gene"])
        jaccard, entered, exited = top_jaccard_from_counts(
            baseline_counts,
            final_counts,
        )
        n_rescued = int(keep_rescue.sum())
        top_gene = str(rescued_counts.index[0]) if len(rescued_counts) else ""
        top_gene_count = int(rescued_counts.iloc[0]) if len(rescued_counts) else 0
        rows.append(
            {
                "gate_name": gate_name,
                "min_score_delta": min_score_delta,
                "min_geomean_prob": min_geomean_prob,
                "max_corrected_round_margin": max_corrected_margin,
                "n_rescued_kept": n_rescued,
                "n_assigned_posthoc": int(is_assigned(posthoc_gene).sum()),
                "match_rate_posthoc": safe_ratio(is_assigned(posthoc_gene).sum(), len(decoded)),
                "top10_jaccard_vs_baseline": jaccard,
                "top10_entered": ";".join(entered),
                "top10_exited": ";".join(exited),
                "top_rescued_gene": top_gene,
                "top_rescued_count": top_gene_count,
                "top_rescued_fraction": safe_ratio(top_gene_count, n_rescued),
                "rescued_median_score_delta": float(score_delta.loc[keep_rescue].median())
                if n_rescued
                else np.nan,
                "rescued_median_geomean_prob": float(
                    geomean_prob.loc[keep_rescue].median()
                )
                if n_rescued
                else np.nan,
                "rescued_median_corrected_round_margin": float(
                    corrected_margin.loc[keep_rescue].median()
                )
                if n_rescued
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


def auto_codebook_args(
    result_dir: Path,
    dataset: str,
    fov_id: str,
) -> tuple[Path | None, int | None]:
    metadata_path = (
        BENCHMARK_ROOT
        / "e2e_backend_comparison"
        / "results"
        / dataset
        / fov_id
        / "python_global_only"
        / "run_metadata.json"
    )
    if metadata_path.exists():
        config = json.loads(metadata_path.read_text())["dataset_config"]
        return Path(config["codebook_path"]), config.get("split_index")

    raw_metadata_path = result_dir / dataset / "input_metadata" / f"{fov_id}.json"
    if raw_metadata_path.exists():
        metadata = json.loads(raw_metadata_path.read_text())
        config = metadata.get("dataset_config", {})
        codebook_path = config.get("codebook_path")
        if codebook_path:
            return Path(codebook_path), config.get("split_index")

    if dataset == "synthetic_medium":
        codebook_path = BENCHMARK_ROOT / "decoding" / "postcode" / "data" / dataset
        return codebook_path / "codebook.csv", None

    if dataset.startswith("synthetic_"):
        preset = dataset.removeprefix("synthetic_")
        codebook_path = BENCHMARK_ROOT / "e2e" / "data" / preset / "codebook.csv"
        return codebook_path, None

    if dataset in RAW_REAL_CODEBOOKS:
        return RAW_REAL_CODEBOOKS[dataset]

    return None, None


def load_seq_to_gene(
    *,
    result_dir: Path,
    dataset: str,
    fov_id: str,
    codebook_path: Path | None,
    split_index: int | None,
) -> dict[str, str] | None:
    from starfinder.barcode import load_codebook

    resolved_path = codebook_path
    resolved_split = split_index
    if resolved_path is None:
        resolved_path, auto_split = auto_codebook_args(result_dir, dataset, fov_id)
        if resolved_split is None:
            resolved_split = auto_split
    if resolved_path is None or not resolved_path.exists():
        return None
    _gene_to_seq, seq_to_gene = load_codebook(
        resolved_path,
        split_index=resolved_split,
    )
    return seq_to_gene


def infer_codebook_shape(seq_to_gene: dict[str, str]) -> tuple[int, int]:
    first_seq = next(iter(seq_to_gene))
    labels = {
        int(char)
        for seq in seq_to_gene
        for char in seq
        if char.isdigit()
    }
    return max(labels), len(first_seq)


def make_codebook_neighbor_bias(
    seq_to_gene: dict[str, str] | None,
    gene_table: pd.DataFrame,
) -> pd.DataFrame:
    if not seq_to_gene:
        return pd.DataFrame()

    from starfinder.barcode.codebook_aware import build_one_error_index

    n_channels, n_rounds = infer_codebook_shape(seq_to_gene)
    one_error_index = build_one_error_index(seq_to_gene, n_channels, n_rounds)
    seq_to_base = {
        seq: strip_pad_suffix(gene)
        for seq, gene in seq_to_gene.items()
    }
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "neighbor_slots": 0,
            "neighbor_observed_sequences": 0,
            "unique_base_neighbors": 0,
            "ambiguous_base_neighbors": 0,
            "unique_sequence_neighbors": 0,
            "ambiguous_sequence_neighbors": 0,
        }
    )

    for observed_seq, candidates in one_error_index.items():
        if observed_seq in seq_to_gene:
            continue
        candidate_bases = [seq_to_base[seq] for seq in candidates]
        distinct_bases = set(candidate_bases)
        for base_gene in distinct_bases:
            n_base_candidates = candidate_bases.count(base_gene)
            stats[base_gene]["neighbor_slots"] += n_base_candidates
            stats[base_gene]["neighbor_observed_sequences"] += 1
            if len(distinct_bases) == 1:
                stats[base_gene]["unique_base_neighbors"] += 1
            else:
                stats[base_gene]["ambiguous_base_neighbors"] += 1
            if len(candidates) == 1:
                stats[base_gene]["unique_sequence_neighbors"] += 1
            else:
                stats[base_gene]["ambiguous_sequence_neighbors"] += 1

    neighbor_table = pd.DataFrame(
        [{"gene_base": gene, **values} for gene, values in stats.items()]
    )
    if neighbor_table.empty:
        return neighbor_table

    count_columns = [
        "gene_base",
        "baseline_count",
        "exact_count",
        "rescued_count",
        "final_count",
        "baseline_rank",
        "final_rank",
    ]
    merged = neighbor_table.merge(
        gene_table[count_columns],
        on="gene_base",
        how="left",
    )
    for column in ["baseline_count", "exact_count", "rescued_count", "final_count"]:
        merged[column] = merged[column].fillna(0).astype(int)
    merged["neighbor_slots_rank"] = rank_series(merged["neighbor_slots"])
    merged["rescued_per_neighbor_slot"] = merged["rescued_count"] / merged[
        "neighbor_slots"
    ]
    merged["rescued_per_unique_base_neighbor"] = merged["rescued_count"] / merged[
        "unique_base_neighbors"
    ]
    return merged.sort_values(
        ["rescued_count", "neighbor_slots", "gene_base"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def parse_corrected_rounds(value: Any) -> list[int]:
    if value is None or pd.isna(value):
        return []
    rounds = []
    for token in str(value).split(","):
        token = token.strip()
        if token == "":
            continue
        rounds.append(int(float(token)))
    return rounds


def make_round_and_transition_tables(
    decoded: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rescue_mask = decoded["call_type"].fillna("").astype(str).str.startswith("rescued")
    round_rows = []
    transition_rows = []
    sequence_rows = []

    for row in decoded.loc[rescue_mask].itertuples(index=False):
        corrected_rounds = parse_corrected_rounds(getattr(row, "corrected_rounds"))
        wta_seq = str(getattr(row, "color_seq_wta"))
        decoded_seq = str(getattr(row, "decoded_seq"))
        gene = strip_pad_suffix(getattr(row, "gene"))
        sequence_rows.append(
            {
                "color_seq_wta": wta_seq,
                "decoded_seq": decoded_seq,
                "gene_base": gene,
                "corrected_rounds": getattr(row, "corrected_rounds"),
            }
        )
        for round_idx in corrected_rounds:
            wta_color = wta_seq[round_idx] if round_idx < len(wta_seq) else ""
            decoded_color = (
                decoded_seq[round_idx] if round_idx < len(decoded_seq) else ""
            )
            common = {
                "round_index_0based": round_idx,
                "round_number_1based": round_idx + 1,
                "gene_base": gene,
                "score_delta": getattr(row, "score_delta"),
                "geomean_prob": getattr(row, "geomean_prob"),
                "corrected_round_margin": getattr(row, "corrected_round_margin"),
            }
            round_rows.append(common)
            transition_rows.append(
                {
                    **common,
                    "wta_color": wta_color,
                    "decoded_color": decoded_color,
                    "transition": f"{wta_color}->{decoded_color}",
                }
            )

    round_frame = pd.DataFrame(round_rows)
    if round_frame.empty:
        round_summary = pd.DataFrame()
    else:
        round_summary = (
            round_frame.groupby(["round_index_0based", "round_number_1based"])
            .agg(
                n_corrections=("round_index_0based", "size"),
                median_score_delta=("score_delta", "median"),
                median_geomean_prob=("geomean_prob", "median"),
                median_corrected_round_margin=("corrected_round_margin", "median"),
            )
            .reset_index()
            .sort_values("round_index_0based")
        )
        round_summary["correction_fraction"] = round_summary[
            "n_corrections"
        ] / round_summary["n_corrections"].sum()

    transition_frame = pd.DataFrame(transition_rows)
    if transition_frame.empty:
        transition_summary = pd.DataFrame()
    else:
        transition_summary = (
            transition_frame.groupby(
                [
                    "round_index_0based",
                    "round_number_1based",
                    "wta_color",
                    "decoded_color",
                    "transition",
                ]
            )
            .agg(
                n_corrections=("transition", "size"),
                median_score_delta=("score_delta", "median"),
                median_geomean_prob=("geomean_prob", "median"),
                median_corrected_round_margin=("corrected_round_margin", "median"),
            )
            .reset_index()
            .sort_values(
                ["n_corrections", "round_index_0based", "transition"],
                ascending=[False, True, True],
            )
        )

    sequence_frame = pd.DataFrame(sequence_rows)
    if sequence_frame.empty:
        sequence_summary = pd.DataFrame()
    else:
        sequence_summary = (
            sequence_frame.groupby(
                ["color_seq_wta", "decoded_seq", "gene_base", "corrected_rounds"]
            )
            .size()
            .reset_index(name="n_rescued")
            .sort_values(["n_rescued", "gene_base"], ascending=[False, True])
        )

    return round_summary, transition_summary, sequence_summary


def add_spatial_features(
    decoded: pd.DataFrame,
    spots: pd.DataFrame,
    *,
    edge_margin_px: float,
) -> pd.DataFrame:
    columns = ["spot_id", "z", "y", "x"]
    merged = decoded.merge(spots[columns], on="spot_id", how="left")

    x_min = float(merged["x"].min())
    x_max = float(merged["x"].max())
    y_min = float(merged["y"].min())
    y_max = float(merged["y"].max())
    z_min = float(merged["z"].min())
    z_max = float(merged["z"].max())

    merged["x_edge"] = (merged["x"] <= x_min + edge_margin_px) | (
        merged["x"] >= x_max - edge_margin_px
    )
    merged["y_edge"] = (merged["y"] <= y_min + edge_margin_px) | (
        merged["y"] >= y_max - edge_margin_px
    )
    merged["z_edge"] = (merged["z"] <= z_min) | (merged["z"] >= z_max)
    merged["xy_edge"] = merged["x_edge"] | merged["y_edge"]
    merged["any_edge"] = merged["xy_edge"] | merged["z_edge"]
    return merged


def make_spatial_tables(
    merged: pd.DataFrame,
    *,
    grid_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = merged.copy()
    work["is_exact"] = work["call_type"] == "exact"
    work["is_rescued"] = work["call_type"].fillna("").astype(str).str.startswith(
        "rescued"
    )
    work["is_assigned"] = is_assigned(work["gene"])

    x_bins = np.linspace(work["x"].min(), work["x"].max(), grid_size + 1)
    y_bins = np.linspace(work["y"].min(), work["y"].max(), grid_size + 1)
    work["x_bin"] = pd.cut(
        work["x"],
        bins=x_bins,
        include_lowest=True,
        labels=False,
        duplicates="drop",
    )
    work["y_bin"] = pd.cut(
        work["y"],
        bins=y_bins,
        include_lowest=True,
        labels=False,
        duplicates="drop",
    )

    grid = (
        work.groupby(["y_bin", "x_bin"], dropna=False)
        .agg(
            n_reads=("spot_id", "size"),
            n_exact=("is_exact", "sum"),
            n_rescued=("is_rescued", "sum"),
            n_assigned=("is_assigned", "sum"),
            median_score_delta=("score_delta", "median"),
            median_geomean_prob=("geomean_prob", "median"),
            median_intensity=("mean_total_intensity", "median"),
            x_min=("x", "min"),
            x_max=("x", "max"),
            y_min=("y", "min"),
            y_max=("y", "max"),
        )
        .reset_index()
    )
    grid["rescue_rate"] = grid["n_rescued"] / grid["n_reads"]
    grid["assigned_rate"] = grid["n_assigned"] / grid["n_reads"]

    z_slices = (
        work.groupby("z", dropna=False)
        .agg(
            n_reads=("spot_id", "size"),
            n_exact=("is_exact", "sum"),
            n_rescued=("is_rescued", "sum"),
            n_assigned=("is_assigned", "sum"),
            median_score_delta=("score_delta", "median"),
            median_geomean_prob=("geomean_prob", "median"),
            median_intensity=("mean_total_intensity", "median"),
        )
        .reset_index()
        .sort_values("z")
    )
    z_slices["rescue_rate"] = z_slices["n_rescued"] / z_slices["n_reads"]
    z_slices["assigned_rate"] = z_slices["n_assigned"] / z_slices["n_reads"]

    edge_rows = []
    for feature in ["x_edge", "y_edge", "z_edge", "xy_edge", "any_edge"]:
        for value, group in work.groupby(feature, dropna=False):
            edge_rows.append(
                {
                    "feature": feature,
                    "value": bool_value(value),
                    "n_reads": int(len(group)),
                    "n_rescued": int(group["is_rescued"].sum()),
                    "n_assigned": int(group["is_assigned"].sum()),
                    "rescue_rate": safe_ratio(group["is_rescued"].sum(), len(group)),
                    "assigned_rate": safe_ratio(group["is_assigned"].sum(), len(group)),
                }
            )
    edge_summary = pd.DataFrame(edge_rows).sort_values(["feature", "value"])

    return grid, z_slices, edge_summary


def make_rescue_deciles(decoded: pd.DataFrame) -> pd.DataFrame:
    rescue_mask = decoded["call_type"].fillna("").astype(str).str.startswith("rescued")
    rescued = decoded.loc[rescue_mask].copy()
    if rescued.empty:
        return pd.DataFrame()

    sort_metric = pd.to_numeric(rescued["score_delta"], errors="coerce")
    rescued["score_delta_decile"] = pd.qcut(
        sort_metric.rank(method="first"),
        10,
        labels=False,
        duplicates="drop",
    )
    rescued["score_delta_decile"] = rescued["score_delta_decile"].astype(int) + 1
    rescued["gene_base"] = rescued["gene"].map(strip_pad_suffix)

    rows = []
    for decile, group in rescued.groupby("score_delta_decile"):
        gene_counts = group["gene_base"].value_counts()
        top_gene = str(gene_counts.index[0]) if len(gene_counts) else ""
        rows.append(
            {
                "score_delta_decile": int(decile),
                "n_rescued": int(len(group)),
                "score_delta_min": float(group["score_delta"].min()),
                "score_delta_max": float(group["score_delta"].max()),
                "median_score_delta": float(group["score_delta"].median()),
                "median_geomean_prob": float(group["geomean_prob"].median()),
                "median_corrected_round_margin": float(
                    group["corrected_round_margin"].median()
                ),
                "median_intensity": float(group["mean_total_intensity"].median()),
                "top_gene": top_gene,
                "top_gene_fraction": safe_ratio(gene_counts.iloc[0], len(group))
                if len(gene_counts)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("score_delta_decile")


def make_rescued_examples(decoded: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rescue_mask = decoded["call_type"].fillna("").astype(str).str.startswith("rescued")
    rescued = decoded.loc[rescue_mask].copy()
    if rescued.empty:
        return pd.DataFrame()
    rescued["gene_base"] = rescued["gene"].map(strip_pad_suffix)

    columns = [
        "spot_id",
        "color_seq_wta",
        "decoded_seq",
        "gene",
        "gene_base",
        "call_type",
        "corrected_rounds",
        "score_delta",
        "geomean_prob",
        "min_round_margin",
        "corrected_round_margin",
        "mean_total_intensity",
    ]
    low = rescued.sort_values(
        ["score_delta", "corrected_round_margin", "geomean_prob"],
        ascending=[True, True, True],
    ).head(top_n)
    high = rescued.sort_values(
        ["score_delta", "corrected_round_margin", "geomean_prob"],
        ascending=[False, False, False],
    ).head(top_n)
    low = low.assign(example_group="lowest_score_delta")
    high = high.assign(example_group="highest_score_delta")
    return pd.concat([low, high], ignore_index=True)[["example_group", *columns]]


def median_lookup(
    confidence_table: pd.DataFrame,
    call_group: str,
    metric: str,
) -> float:
    match = confidence_table[
        (confidence_table["call_group"] == call_group)
        & (confidence_table["metric"] == metric)
    ]
    if match.empty or "median" not in match:
        return float("nan")
    return float(match.iloc[0]["median"])


def edge_rate(edge_summary: pd.DataFrame, feature: str, value: bool) -> float:
    match = edge_summary[
        (edge_summary["feature"] == feature) & (edge_summary["value"] == value)
    ]
    if match.empty:
        return float("nan")
    return float(match.iloc[0]["rescue_rate"])


def build_summary(
    *,
    dataset: str,
    fov_id: str,
    config: str,
    baseline_config: str,
    decoded: pd.DataFrame,
    baseline: pd.DataFrame,
    gene_table: pd.DataFrame,
    top_jaccard_value: float,
    top_entered: list[str],
    top_exited: list[str],
    round_summary: pd.DataFrame,
    transition_summary: pd.DataFrame,
    sequence_summary: pd.DataFrame,
    confidence_table: pd.DataFrame,
    grid: pd.DataFrame,
    edge_summary: pd.DataFrame,
    codebook_bias: pd.DataFrame,
    min_bin_reads: int,
) -> dict[str, Any]:
    call_values = decoded["call_type"].fillna("").astype(str)
    rescue_mask = call_values.str.startswith("rescued")
    exact_mask = call_values == "exact"
    assigned = is_assigned(decoded["gene"])
    baseline_assigned = is_assigned(baseline["gene"])
    n_rescued = int(rescue_mask.sum())
    n_input = int(len(decoded))

    top_rescued = gene_table.sort_values(
        ["rescued_count", "gene_base"],
        ascending=[False, True],
    ).head(10)
    top_rescue_gene = (
        str(top_rescued.iloc[0]["gene_base"]) if len(top_rescued) else ""
    )
    top_rescue_count = int(top_rescued.iloc[0]["rescued_count"]) if len(top_rescued) else 0
    top_rescue_fraction = safe_ratio(top_rescue_count, n_rescued)
    top10_rescue_fraction = safe_ratio(top_rescued["rescued_count"].sum(), n_rescued)

    dominant_round = None
    dominant_round_fraction = float("nan")
    if not round_summary.empty:
        top_round = round_summary.sort_values(
            ["n_corrections", "round_index_0based"],
            ascending=[False, True],
        ).iloc[0]
        dominant_round = int(top_round["round_number_1based"])
        dominant_round_fraction = float(top_round["correction_fraction"])

    dominant_transition = ""
    dominant_transition_fraction = float("nan")
    if not transition_summary.empty:
        top_transition = transition_summary.iloc[0]
        dominant_transition = (
            f"round{int(top_transition['round_number_1based'])}:"
            f"{top_transition['transition']}"
        )
        dominant_transition_fraction = safe_ratio(
            top_transition["n_corrections"],
            transition_summary["n_corrections"].sum(),
        )

    top_sequence_fraction = float("nan")
    if not sequence_summary.empty:
        top_sequence_fraction = safe_ratio(sequence_summary.iloc[0]["n_rescued"], n_rescued)

    exact_intensity_median = median_lookup(
        confidence_table,
        "exact",
        "mean_total_intensity",
    )
    rescued_intensity_median = median_lookup(
        confidence_table,
        "rescued",
        "mean_total_intensity",
    )
    exact_geomean_median = median_lookup(confidence_table, "exact", "geomean_prob")
    rescued_geomean_median = median_lookup(
        confidence_table,
        "rescued",
        "geomean_prob",
    )
    rescued_score_delta_median = median_lookup(
        confidence_table,
        "rescued",
        "score_delta",
    )
    rescued_corrected_margin_median = median_lookup(
        confidence_table,
        "rescued",
        "corrected_round_margin",
    )

    interior_rate = edge_rate(edge_summary, "xy_edge", False)
    edge_xy_rate = edge_rate(edge_summary, "xy_edge", True)
    edge_enrichment = safe_ratio(edge_xy_rate, interior_rate)
    global_rescue_rate = safe_ratio(n_rescued, n_input)
    eligible_bins = grid.loc[grid["n_reads"] >= min_bin_reads]
    max_bin = eligible_bins.sort_values(
        ["rescue_rate", "n_reads"],
        ascending=[False, False],
    ).head(1)
    if max_bin.empty:
        max_bin_rate = float("nan")
        max_bin_multiplier = float("nan")
        max_bin_reads = 0
        max_bin_xy = None
    else:
        max_bin_rate = float(max_bin.iloc[0]["rescue_rate"])
        max_bin_multiplier = safe_ratio(max_bin_rate, global_rescue_rate)
        max_bin_reads = int(max_bin.iloc[0]["n_reads"])
        max_bin_xy = [
            int(max_bin.iloc[0]["x_bin"]),
            int(max_bin.iloc[0]["y_bin"]),
        ]

    warnings = []
    if top_rescue_fraction > 0.10:
        warnings.append("rescues_concentrated_in_one_gene")
    if top10_rescue_fraction > 0.50:
        warnings.append("rescues_concentrated_in_top10_genes")
    if dominant_round_fraction > 0.35:
        warnings.append("rescues_concentrated_in_one_round")
    if dominant_transition_fraction > 0.20:
        warnings.append("rescues_concentrated_in_one_channel_transition")
    if top_sequence_fraction > 0.05:
        warnings.append("rescues_concentrated_in_one_wta_sequence")
    if not np.isnan(edge_enrichment) and edge_enrichment > 2.0:
        warnings.append("rescues_enriched_near_xy_edges")
    if not np.isnan(max_bin_multiplier) and max_bin_multiplier > 3.0:
        warnings.append("rescues_concentrated_in_spatial_hotspot")
    if (
        not np.isnan(rescued_intensity_median)
        and not np.isnan(exact_intensity_median)
        and rescued_intensity_median < 0.5 * exact_intensity_median
    ):
        warnings.append("rescues_have_low_intensity_vs_exact")
    if (
        not np.isnan(rescued_geomean_median)
        and not np.isnan(exact_geomean_median)
        and rescued_geomean_median < 0.8 * exact_geomean_median
    ):
        warnings.append("rescues_have_low_probability_vs_exact")
    if top_jaccard_value < 0.9:
        warnings.append("top_gene_set_changed")

    codebook_neighbor_rescue_spearman = float("nan")
    top_rescued_gene_neighbor_slots = None
    top_rescued_gene_neighbor_rank = None
    top_rescued_gene_rescued_per_neighbor = float("nan")
    if not codebook_bias.empty:
        if (
            codebook_bias["rescued_count"].nunique() > 1
            and codebook_bias["neighbor_slots"].nunique() > 1
        ):
            codebook_neighbor_rescue_spearman = float(
                codebook_bias["rescued_count"].corr(
                    codebook_bias["neighbor_slots"],
                    method="spearman",
                )
            )
        top_bias = codebook_bias.loc[codebook_bias["gene_base"] == top_rescue_gene]
        if not top_bias.empty:
            top_rescued_gene_neighbor_slots = int(top_bias.iloc[0]["neighbor_slots"])
            top_rescued_gene_neighbor_rank = int(
                top_bias.iloc[0]["neighbor_slots_rank"]
            )
            top_rescued_gene_rescued_per_neighbor = float(
                top_bias.iloc[0]["rescued_per_neighbor_slot"]
            )
        if (
            not np.isnan(codebook_neighbor_rescue_spearman)
            and codebook_neighbor_rescue_spearman > 0.7
        ):
            warnings.append("rescues_follow_codebook_neighbor_bias")

    return {
        "dataset": dataset,
        "fov_id": fov_id,
        "config": config,
        "baseline_config": baseline_config,
        "n_input": n_input,
        "n_exact": int(exact_mask.sum()),
        "n_rescued": n_rescued,
        "n_assigned": int(assigned.sum()),
        "baseline_n_assigned": int(baseline_assigned.sum()),
        "rescue_rate_input": safe_ratio(n_rescued, n_input),
        "rescue_rate_assigned": safe_ratio(n_rescued, assigned.sum()),
        "assigned_gain_vs_baseline": int(assigned.sum() - baseline_assigned.sum()),
        "top_rescued_gene": top_rescue_gene,
        "top_rescued_count": top_rescue_count,
        "top_rescued_fraction": top_rescue_fraction,
        "top10_rescued_fraction": top10_rescue_fraction,
        "top_gene_jaccard": top_jaccard_value,
        "top_gene_entered": top_entered,
        "top_gene_exited": top_exited,
        "dominant_corrected_round_1based": dominant_round,
        "dominant_corrected_round_fraction": dominant_round_fraction,
        "dominant_transition": dominant_transition,
        "dominant_transition_fraction": dominant_transition_fraction,
        "top_wta_to_decoded_sequence_fraction": top_sequence_fraction,
        "exact_median_intensity": exact_intensity_median,
        "rescued_median_intensity": rescued_intensity_median,
        "exact_median_geomean_prob": exact_geomean_median,
        "rescued_median_geomean_prob": rescued_geomean_median,
        "rescued_median_score_delta": rescued_score_delta_median,
        "rescued_median_corrected_round_margin": rescued_corrected_margin_median,
        "xy_edge_rescue_rate": edge_xy_rate,
        "xy_interior_rescue_rate": interior_rate,
        "xy_edge_rescue_enrichment": edge_enrichment,
        "global_rescue_rate": global_rescue_rate,
        "max_grid_bin_rescue_rate": max_bin_rate,
        "max_grid_bin_rescue_rate_multiplier": max_bin_multiplier,
        "max_grid_bin_reads": max_bin_reads,
        "max_grid_bin_xy": max_bin_xy,
        "codebook_neighbor_rescue_spearman": codebook_neighbor_rescue_spearman,
        "top_rescued_gene_neighbor_slots": top_rescued_gene_neighbor_slots,
        "top_rescued_gene_neighbor_rank": top_rescued_gene_neighbor_rank,
        "top_rescued_gene_rescued_per_neighbor_slot": (
            top_rescued_gene_rescued_per_neighbor
        ),
        "artifact_warnings": warnings,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def diagnose_fov(
    *,
    result_dir: Path,
    dataset: str,
    config: str,
    baseline_config: str,
    fov_id: str,
    grid_size: int,
    edge_margin_px: float,
    min_bin_reads: int,
    top_n: int,
    codebook_path: Path | None,
    split_index: int | None,
) -> dict[str, Any]:
    decoded, baseline, spots = read_inputs(
        result_dir,
        dataset,
        config,
        baseline_config,
        fov_id,
    )
    output_dir = result_dir / dataset / "diagnostics" / f"{fov_id}_{config}"
    output_dir.mkdir(parents=True, exist_ok=True)

    gene_table = make_gene_rescue_enrichment(decoded, baseline)
    seq_to_gene = load_seq_to_gene(
        result_dir=result_dir,
        dataset=dataset,
        fov_id=fov_id,
        codebook_path=codebook_path,
        split_index=split_index,
    )
    codebook_bias = make_codebook_neighbor_bias(seq_to_gene, gene_table)
    top_shift, top_jaccard_value, top_entered, top_exited = make_top_shift(
        gene_table,
        top_n=10,
    )
    round_summary, transition_summary, sequence_summary = (
        make_round_and_transition_tables(decoded)
    )
    confidence_table = summarize_confidence(decoded)
    gate_sensitivity = make_gate_sensitivity(decoded, baseline)
    merged = add_spatial_features(
        decoded,
        spots,
        edge_margin_px=edge_margin_px,
    )
    spatial_grid, z_slices, edge_summary = make_spatial_tables(
        merged,
        grid_size=grid_size,
    )
    rescue_deciles = make_rescue_deciles(decoded)
    rescued_examples = make_rescued_examples(decoded, top_n=top_n)
    summary = build_summary(
        dataset=dataset,
        fov_id=fov_id,
        config=config,
        baseline_config=baseline_config,
        decoded=decoded,
        baseline=baseline,
        gene_table=gene_table,
        top_jaccard_value=top_jaccard_value,
        top_entered=top_entered,
        top_exited=top_exited,
        round_summary=round_summary,
        transition_summary=transition_summary,
        sequence_summary=sequence_summary,
        confidence_table=confidence_table,
        grid=spatial_grid,
        edge_summary=edge_summary,
        codebook_bias=codebook_bias,
        min_bin_reads=min_bin_reads,
    )

    gene_table.to_csv(output_dir / "gene_rescue_enrichment.csv", index=False)
    if not codebook_bias.empty:
        codebook_bias.to_csv(output_dir / "codebook_neighbor_bias.csv", index=False)
    top_shift.to_csv(output_dir / "top10_shift.csv", index=False)
    round_summary.to_csv(output_dir / "round_corrections.csv", index=False)
    transition_summary.to_csv(output_dir / "correction_transitions.csv", index=False)
    sequence_summary.to_csv(output_dir / "rescue_sequence_patterns.csv", index=False)
    confidence_table.to_csv(output_dir / "confidence_by_call_type.csv", index=False)
    gate_sensitivity.to_csv(output_dir / "gate_sensitivity.csv", index=False)
    rescue_deciles.to_csv(output_dir / "rescued_confidence_deciles.csv", index=False)
    spatial_grid.to_csv(output_dir / "spatial_bins.csv", index=False)
    z_slices.to_csv(output_dir / "z_slices.csv", index=False)
    edge_summary.to_csv(output_dir / "edge_summary.csv", index=False)
    rescued_examples.to_csv(output_dir / "rescued_examples.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_safe)
    )
    return summary


def discover_fovs(result_dir: Path, dataset: str, config: str) -> list[str]:
    config_dir = result_dir / dataset / "codebook_aware" / config
    return sorted(path.stem for path in config_dir.glob("*.csv"))


def main() -> None:
    args = parse_args()
    fovs = args.fovs or discover_fovs(args.result_dir, args.dataset, args.config)
    if not fovs:
        raise SystemExit(
            f"No decoded CSVs found for {args.dataset}/{args.config} under "
            f"{args.result_dir}"
        )

    summaries = []
    for fov_id in fovs:
        summary = diagnose_fov(
            result_dir=args.result_dir,
            dataset=args.dataset,
            config=args.config,
            baseline_config=args.baseline_config,
            fov_id=fov_id,
            grid_size=args.grid_size,
            edge_margin_px=args.edge_margin_px,
            min_bin_reads=args.min_bin_reads,
            top_n=args.top_n,
            codebook_path=args.codebook_path,
            split_index=args.split_index,
        )
        summaries.append(summary)
        print(
            f"{args.dataset} {fov_id} {args.config}: "
            f"rescued={summary['n_rescued']:,} "
            f"top_gene={summary['top_rescued_gene']} "
            f"top_fraction={summary['top_rescued_fraction']:.4f} "
            f"top10_jaccard={summary['top_gene_jaccard']:.4f} "
            f"warnings={','.join(summary['artifact_warnings']) or 'none'}"
        )

    summary_path = args.result_dir / args.dataset / "diagnostics" / (
        f"{args.config}_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2, default=_json_safe))
    print(f"Saved diagnostics under {summary_path.parent}")


if __name__ == "__main__":
    main()
