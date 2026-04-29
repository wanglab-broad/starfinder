#!/usr/bin/env python3
"""Generate image-evidence QC for codebook-aware rescued reads."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from starfinder.barcode import load_codebook
from starfinder.barcode.codebook_aware import (
    build_one_error_index,
    candidate_sequences,
    channel_probabilities,
    score_candidates,
)

BENCHMARK_ROOT = Path("/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark")
DEFAULT_RESULT_DIR = BENCHMARK_ROOT / "decoding" / "codebook_aware" / "results"
PAD_SUFFIX_RE = re.compile(r"_(STAR|RIBO)_pad_\d+$")
SEQ_COLUMNS = ["color_seq_wta", "decoded_seq", "corrected_rounds"]
RAW_REAL_DATASETS: dict[str, dict[str, Any]] = {
    "cell_culture_3D": {
        "data_root": Path(
            "/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/cell-culture-3D"
        ),
        "codebook_path": Path(
            "/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/cell-culture-3D/genes.csv"
        ),
        "sample_id": "cell-culture-3D",
        "n_rounds": 6,
        "ref_round": "round1",
        "channel_order": ["ch00", "ch02", "ch01", "ch03"],
        "fov_pattern": "Position%03d",
        "rotate_angle": -90,
        "snr_threshold": 5.0,
        "split_index": None,
    },
    "tissue_2D": {
        "data_root": Path(
            "/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/tissue-2D"
        ),
        "codebook_path": Path(
            "/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/tissue-2D/genes.csv"
        ),
        "sample_id": "tissue-2D",
        "n_rounds": 4,
        "ref_round": "round1",
        "channel_order": ["ch00", "ch02", "ch01", "ch03"],
        "fov_pattern": "tile_%d",
        "rotate_angle": -90,
        "snr_threshold": 5.0,
        "split_index": None,
    },
}


@dataclass
class StackCache:
    stack_dir: Path
    handles: dict[int, tifffile.TiffFile]

    def get_plane(self, round_idx: int, z_idx: int) -> np.ndarray:
        if round_idx not in self.handles:
            path = self.stack_dir / f"round{round_idx + 1}.tif"
            if not path.exists():
                raise FileNotFoundError(path)
            self.handles[round_idx] = tifffile.TiffFile(path)
        tif = self.handles[round_idx]
        page = tif.pages[int(z_idx)]
        plane = page.asarray()
        if plane.ndim != 3:
            raise ValueError(
                f"Expected z-plane with shape (Y, X, C), got {plane.shape}"
            )
        return plane

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--dataset", default="aging")
    parser.add_argument("--fov", default="Position400")
    parser.add_argument("--config", default="balanced")
    parser.add_argument("--genes", nargs="+", default=["Flt3", "Mbp"])
    parser.add_argument("--control-genes", nargs="+", default=["Penk", "Camk2a"])
    parser.add_argument("--examples-per-group", type=int, default=6)
    parser.add_argument("--patch-radius", type=int, default=18)
    parser.add_argument("--all-round-limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--codebook-path", type=Path, default=None)
    parser.add_argument("--split-index", type=int, default=None)
    parser.add_argument(
        "--tensor-metrics-only",
        action="store_true",
        help="Only write all-rescued_h1 tensor metrics; skip montage generation.",
    )
    return parser.parse_args()


def strip_pad_suffix(gene_name: Any) -> str | None:
    if gene_name is None or pd.isna(gene_name):
        return None
    value = str(gene_name)
    if value == "":
        return None
    return PAD_SUFFIX_RE.sub("", value)


def normalize_sequence(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def parse_rounds(value: Any) -> list[int]:
    if value is None or pd.isna(value):
        return []
    rounds = []
    for token in str(value).split(","):
        token = token.strip()
        if token == "":
            continue
        rounds.append(int(float(token)))
    return rounds


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_decoded(result_dir: Path, dataset: str, config: str, fov_id: str) -> pd.DataFrame:
    path = result_dir / dataset / "codebook_aware" / config / f"{fov_id}.csv"
    dtype = {column: "string" for column in SEQ_COLUMNS}
    dtype.update(
        {
            "gene": "string",
            "gene_wta": "string",
            "call_type": "string",
            "reject_reason": "string",
        }
    )
    decoded = pd.read_csv(path, dtype=dtype, low_memory=False)
    for column in ["color_seq_wta", "decoded_seq"]:
        decoded[column] = decoded[column].map(normalize_sequence)
    decoded["gene_base"] = decoded["gene"].map(strip_pad_suffix)
    decoded["gene_wta_base"] = decoded["gene_wta"].map(strip_pad_suffix)
    decoded["call_group"] = np.select(
        [
            decoded["call_type"].fillna("").astype(str).eq("exact"),
            decoded["call_type"].fillna("").astype(str).str.startswith("rescued"),
            decoded["call_type"].fillna("").astype(str).eq("no_call"),
        ],
        ["exact", "rescued", "no_call"],
        default=decoded["call_type"].fillna("").astype(str),
    )
    for column in [
        "score",
        "score_delta",
        "geomean_prob",
        "min_round_margin",
        "corrected_round_margin",
        "mean_total_intensity",
    ]:
        decoded[column] = pd.to_numeric(decoded[column], errors="coerce")
    return decoded


def read_spots(result_dir: Path, dataset: str, fov_id: str) -> pd.DataFrame:
    path = result_dir / dataset / "spots" / f"{fov_id}.csv"
    return pd.read_csv(path, usecols=["spot_id", "z", "y", "x"], low_memory=False)


def load_tensor(result_dir: Path, dataset: str, fov_id: str) -> np.ndarray:
    path = result_dir / dataset / "intensity_tensors" / f"{fov_id}.npy"
    return np.load(path, mmap_mode="r")


def backend_variant_dir(dataset: str, fov_id: str) -> Path:
    return (
        BENCHMARK_ROOT
        / "e2e_backend_comparison"
        / "results"
        / dataset
        / fov_id
        / "python_global_only"
    )


def raw_input_metadata_path(result_dir: Path, dataset: str, fov_id: str) -> Path:
    return result_dir / dataset / "input_metadata" / f"{fov_id}.json"


def raw_codebook_args(
    result_dir: Path,
    dataset: str,
    fov_id: str,
) -> tuple[Path, int | None] | None:
    metadata_path = raw_input_metadata_path(result_dir, dataset, fov_id)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        config = metadata.get("dataset_config", {})
        codebook_path = config.get("codebook_path")
        if codebook_path:
            return Path(codebook_path), config.get("split_index")

    if dataset in RAW_REAL_DATASETS:
        config = RAW_REAL_DATASETS[dataset]
        return Path(config["codebook_path"]), config.get("split_index")

    return None


def raw_registered_stack_dir(result_dir: Path, dataset: str, fov_id: str) -> Path:
    return result_dir / dataset / "registered_stacks" / fov_id


def generate_raw_registered_stack_cache(
    result_dir: Path,
    dataset: str,
    fov_id: str,
) -> Path:
    if dataset not in RAW_REAL_DATASETS:
        raise FileNotFoundError(f"No raw-E2E stack cache recipe for {dataset}/{fov_id}")

    from starfinder.dataset import STARMapDataset
    from starfinder.dataset.types import LayerState
    from starfinder.io import save_stack

    config = RAW_REAL_DATASETS[dataset]
    stack_dir = raw_registered_stack_dir(result_dir, dataset, fov_id)
    expected = [
        stack_dir / f"round{round_idx}.tif"
        for round_idx in range(1, int(config["n_rounds"]) + 1)
    ]
    if all(path.exists() for path in expected):
        return stack_dir

    ds = STARMapDataset(
        input_root=config["data_root"],
        output_root=result_dir / dataset / "registered_stack_cache_output",
        dataset_id="sample-dataset",
        sample_id=config["sample_id"],
        output_id=f"{dataset}-registered-stack-cache",
        layers=LayerState(
            seq=[f"round{i}" for i in range(1, int(config["n_rounds"]) + 1)],
            ref=config["ref_round"],
        ),
        channel_order=config["channel_order"],
        fov_pattern=config["fov_pattern"],
    )
    fov = ds.fov(fov_id)
    print(f"Generating registered stack cache for {dataset} {fov_id}")
    fov.load_raw_images()
    fov.rotate(angle=float(config["rotate_angle"]))
    fov.enhance_contrast(snr_threshold=float(config["snr_threshold"]))
    fov.global_registration()

    stack_dir.mkdir(parents=True, exist_ok=True)
    for round_idx in range(1, int(config["n_rounds"]) + 1):
        round_name = f"round{round_idx}"
        save_stack(fov.images[round_name], stack_dir / f"{round_name}.tif")

    metadata = {
        "dataset": dataset,
        "fov_id": fov_id,
        "source": "raw_e2e_registered_stack_cache",
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in config.items()
        },
        "global_shifts": {
            round_name: list(shift)
            for round_name, shift in fov.global_shifts.items()
        },
    }
    (stack_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return stack_dir


def stack_dir_for(
    dataset: str,
    fov_id: str,
    result_dir: Path = DEFAULT_RESULT_DIR,
) -> Path:
    variant_dir = backend_variant_dir(dataset, fov_id)
    stack_dir = variant_dir / "registered_final"
    if not stack_dir.exists():
        stack_dir = variant_dir / "registered_global"
    if not stack_dir.exists():
        raw_stack_dir = raw_registered_stack_dir(result_dir, dataset, fov_id)
        if raw_stack_dir.exists():
            return raw_stack_dir
        return generate_raw_registered_stack_cache(result_dir, dataset, fov_id)
    return stack_dir


def codebook_args(
    dataset: str,
    fov_id: str,
    explicit_path: Path | None,
    explicit_split: int | None,
    result_dir: Path = DEFAULT_RESULT_DIR,
) -> tuple[Path, int | None]:
    if explicit_path is not None:
        return explicit_path, explicit_split
    metadata_path = backend_variant_dir(dataset, fov_id) / "run_metadata.json"
    if metadata_path.exists():
        config = json.loads(metadata_path.read_text())["dataset_config"]
        return Path(config["codebook_path"]), config.get("split_index")
    raw_args = raw_codebook_args(result_dir, dataset, fov_id)
    if raw_args is not None:
        return raw_args
    if dataset == "synthetic_medium":
        return (
            BENCHMARK_ROOT / "decoding" / "postcode" / "data" / dataset / "codebook.csv",
            None,
        )
    if dataset.startswith("synthetic_"):
        preset = dataset.removeprefix("synthetic_")
        return BENCHMARK_ROOT / "e2e" / "data" / preset / "codebook.csv", None
    raise FileNotFoundError(f"Could not infer codebook path for {dataset}/{fov_id}")


def sample_rows(
    frame: pd.DataFrame,
    *,
    n: int,
    category: str,
    seed: int,
    sort_columns: list[str] | None = None,
    ascending: list[bool] | bool = True,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if sort_columns:
        sampled = frame.sort_values(sort_columns, ascending=ascending).head(n)
    elif len(frame) > n:
        sampled = frame.sample(n=n, random_state=seed)
    else:
        sampled = frame
    sampled = sampled.copy()
    sampled["category"] = category
    return sampled


def target_no_call_candidates(
    decoded: pd.DataFrame,
    seq_to_gene: dict[str, str],
    target_gene: str,
    n_channels: int,
    n_rounds: int,
) -> pd.Series:
    one_error_index = build_one_error_index(seq_to_gene, n_channels, n_rounds)
    seq_to_base = {seq: strip_pad_suffix(gene) for seq, gene in seq_to_gene.items()}
    cache: dict[str, str] = {}
    unique_wta = decoded.loc[decoded["call_group"] == "no_call", "color_seq_wta"].unique()
    for wta_seq in unique_wta:
        seq = normalize_sequence(wta_seq)
        candidates = candidate_sequences(seq, one_error_index, seq_to_gene)
        target_candidates = [
            candidate for candidate in candidates if seq_to_base[candidate] == target_gene
        ]
        cache[seq] = ";".join(target_candidates)
    return decoded["color_seq_wta"].map(cache).fillna("")


def select_examples(
    decoded: pd.DataFrame,
    *,
    seq_to_gene: dict[str, str],
    n_channels: int,
    n_rounds: int,
    target_genes: list[str],
    control_genes: list[str],
    examples_per_group: int,
    seed: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    rescued_h1 = decoded["call_type"].fillna("").astype(str).str.startswith("rescued_h")
    rescued_unknown = decoded["call_type"].fillna("").astype(str).eq("rescued_unknown")
    exact = decoded["call_group"] == "exact"

    for gene in target_genes:
        gene_rescued_h1 = decoded.loc[rescued_h1 & (decoded["gene_base"] == gene)]
        gene_rescued_unknown = decoded.loc[
            rescued_unknown & (decoded["gene_base"] == gene)
        ]
        frames.append(
            sample_rows(
                gene_rescued_h1,
                n=examples_per_group,
                category=f"{gene}_rescued_h1_high_conf",
                seed=seed,
                sort_columns=[
                    "score_delta",
                    "geomean_prob",
                    "corrected_round_margin",
                ],
                ascending=[False, False, True],
            )
        )
        frames.append(
            sample_rows(
                gene_rescued_h1,
                n=examples_per_group,
                category=f"{gene}_rescued_h1_low_conf",
                seed=seed,
                sort_columns=[
                    "score_delta",
                    "geomean_prob",
                    "corrected_round_margin",
                ],
                ascending=[True, True, False],
            )
        )
        frames.append(
            sample_rows(
                gene_rescued_unknown,
                n=examples_per_group,
                category=f"{gene}_rescued_unknown",
                seed=seed,
                sort_columns=["geomean_prob", "mean_total_intensity"],
                ascending=[False, False],
            )
        )
        frames.append(
            sample_rows(
                decoded.loc[exact & (decoded["gene_base"] == gene)],
                n=examples_per_group,
                category=f"{gene}_exact_random",
                seed=seed,
            )
        )

        candidates = target_no_call_candidates(
            decoded,
            seq_to_gene,
            gene,
            n_channels,
            n_rounds,
        )
        no_call_mask = decoded["call_group"] == "no_call"
        no_call_near = decoded.loc[no_call_mask & candidates.astype(bool)].copy()
        no_call_near["target_candidate_seqs"] = candidates.loc[no_call_near.index]
        frames.append(
            sample_rows(
                no_call_near,
                n=examples_per_group,
                category=f"{gene}_near_no_call",
                seed=seed,
                sort_columns=["min_round_margin", "mean_total_intensity"],
                ascending=[True, False],
            )
        )

    for gene in control_genes:
        gene_rescued_h1 = decoded.loc[rescued_h1 & (decoded["gene_base"] == gene)]
        gene_rescued_unknown = decoded.loc[
            rescued_unknown & (decoded["gene_base"] == gene)
        ]
        frames.append(
            sample_rows(
                gene_rescued_h1,
                n=examples_per_group,
                category=f"{gene}_rescued_h1_high_conf",
                seed=seed,
                sort_columns=[
                    "score_delta",
                    "geomean_prob",
                    "corrected_round_margin",
                ],
                ascending=[False, False, True],
            )
        )
        frames.append(
            sample_rows(
                gene_rescued_unknown,
                n=examples_per_group,
                category=f"{gene}_rescued_unknown",
                seed=seed,
                sort_columns=["geomean_prob", "mean_total_intensity"],
                ascending=[False, False],
            )
        )
        frames.append(
            sample_rows(
                decoded.loc[exact & (decoded["gene_base"] == gene)],
                n=examples_per_group,
                category=f"{gene}_exact_random",
                seed=seed,
            )
        )

    frames.append(
        sample_rows(
            decoded.loc[rescued_h1],
            n=examples_per_group,
            category="global_rescued_h1_high_conf",
            seed=seed,
            sort_columns=["score_delta", "geomean_prob", "corrected_round_margin"],
            ascending=[False, False, True],
        )
    )
    frames.append(
        sample_rows(
            decoded.loc[rescued_h1],
            n=examples_per_group,
            category="global_rescued_h1_low_conf",
            seed=seed,
            sort_columns=["score_delta", "geomean_prob", "corrected_round_margin"],
            ascending=[True, True, False],
        )
    )
    frames.append(
        sample_rows(
            decoded.loc[rescued_unknown],
            n=examples_per_group,
            category="global_rescued_unknown",
            seed=seed,
            sort_columns=["geomean_prob", "mean_total_intensity"],
            ascending=[False, False],
        )
    )

    selected = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    selected.insert(0, "example_id", np.arange(len(selected), dtype=int))
    return selected


def probs_for_spot(tensor: np.ndarray, spot_id: int) -> np.ndarray:
    values = np.asarray(tensor[spot_id : spot_id + 1], dtype=np.float64)
    return channel_probabilities(values)[0]


def best_target_candidate(
    probs: np.ndarray,
    candidate_list: Iterable[str],
) -> str:
    candidates = [candidate for candidate in candidate_list if candidate]
    if not candidates:
        return ""
    scores = score_candidates(probs, sorted(set(candidates)))
    return str(scores.iloc[0]["seq"])


def margin_round(probs: np.ndarray) -> int:
    order = np.sort(probs, axis=0)
    margins = order[-1, :] - order[-2, :]
    return int(np.nanargmin(margins))


def hamming_rounds(wta_seq: str, candidate_seq: str) -> list[int]:
    return [
        idx
        for idx, (observed, candidate) in enumerate(zip(wta_seq, candidate_seq))
        if observed.isdigit() and observed != candidate
    ]


def annotate_example(
    row: pd.Series,
    *,
    tensor: np.ndarray,
    seq_to_gene: dict[str, str],
) -> dict[str, Any]:
    spot_id = int(row["spot_id"])
    probs = probs_for_spot(tensor, spot_id)
    intensities = np.asarray(tensor[spot_id], dtype=np.float64)
    wta_seq = normalize_sequence(row["color_seq_wta"])
    decoded_seq = normalize_sequence(row["decoded_seq"])

    target_seq = decoded_seq
    target_gene = row.get("gene_base")
    if not target_seq and str(row.get("target_candidate_seqs", "")):
        target_seq = best_target_candidate(
            probs,
            str(row["target_candidate_seqs"]).split(";"),
        )
        target_gene = strip_pad_suffix(seq_to_gene.get(target_seq, ""))

    corrected_rounds = parse_rounds(row.get("corrected_rounds", ""))
    if corrected_rounds:
        visual_round = corrected_rounds[0]
    elif target_seq:
        mismatches = hamming_rounds(wta_seq, target_seq)
        visual_round = mismatches[0] if mismatches else margin_round(probs)
    else:
        visual_round = margin_round(probs)

    target_color = (
        target_seq[visual_round]
        if target_seq and visual_round < len(target_seq)
        else ""
    )
    wta_color = wta_seq[visual_round] if visual_round < len(wta_seq) else ""
    target_ch = int(target_color) - 1 if target_color.isdigit() else None
    wta_ch = int(wta_color) - 1 if wta_color.isdigit() else None

    round_intensities = intensities[:, visual_round]
    round_probs = probs[:, visual_round]
    top_channel = int(np.argmax(round_probs))
    target_intensity = round_intensities[target_ch] if target_ch is not None else np.nan
    wta_intensity = round_intensities[wta_ch] if wta_ch is not None else np.nan
    target_prob = round_probs[target_ch] if target_ch is not None else np.nan
    wta_prob = round_probs[wta_ch] if wta_ch is not None else np.nan

    record = row.to_dict()
    record.update(
        {
            "target_gene_base": target_gene,
            "target_seq": target_seq,
            "visual_round_0based": visual_round,
            "visual_round_1based": visual_round + 1,
            "visual_wta_color": wta_color,
            "visual_target_color": target_color,
            "visual_top_color": str(top_channel + 1),
            "visual_target_intensity": float(target_intensity),
            "visual_wta_intensity": float(wta_intensity),
            "visual_target_prob": float(target_prob),
            "visual_wta_prob": float(wta_prob),
            "visual_target_to_wta_intensity_ratio": (
                float(target_intensity / wta_intensity)
                if np.isfinite(target_intensity) and wta_intensity > 0
                else np.nan
            ),
            "visual_target_prob_minus_wta_prob": (
                float(target_prob - wta_prob)
                if np.isfinite(target_prob) and np.isfinite(wta_prob)
                else np.nan
            ),
        }
    )

    for round_idx in range(intensities.shape[1]):
        for channel_idx in range(intensities.shape[0]):
            label = f"r{round_idx + 1}_c{channel_idx + 1}"
            record[f"{label}_intensity"] = float(intensities[channel_idx, round_idx])
            record[f"{label}_prob"] = float(probs[channel_idx, round_idx])
    return record


def all_rescued_h1_tensor_metrics(
    decoded: pd.DataFrame,
    tensor: np.ndarray,
    genes: list[str],
) -> pd.DataFrame:
    h1_mask = decoded["call_type"].fillna("").astype(str).str.startswith("rescued_h")
    selected = decoded.loc[h1_mask & decoded["gene_base"].isin(genes)].copy()
    rows = []
    for row in selected.itertuples(index=False):
        corrected_rounds = parse_rounds(getattr(row, "corrected_rounds", ""))
        if not corrected_rounds:
            continue
        spot_id = int(getattr(row, "spot_id"))
        round_idx = corrected_rounds[0]
        wta_seq = normalize_sequence(getattr(row, "color_seq_wta"))
        target_seq = normalize_sequence(getattr(row, "decoded_seq"))
        if round_idx >= len(wta_seq) or round_idx >= len(target_seq):
            continue
        wta_color = wta_seq[round_idx]
        target_color = target_seq[round_idx]
        if not wta_color.isdigit() or not target_color.isdigit():
            continue

        wta_ch = int(wta_color) - 1
        target_ch = int(target_color) - 1
        values = np.asarray(tensor[spot_id : spot_id + 1], dtype=np.float64)
        probs = channel_probabilities(values)[0]
        intensities = values[0]

        wta_intensity = float(intensities[wta_ch, round_idx])
        target_intensity = float(intensities[target_ch, round_idx])
        wta_prob = float(probs[wta_ch, round_idx])
        target_prob = float(probs[target_ch, round_idx])
        rows.append(
            {
                "spot_id": spot_id,
                "gene_base": getattr(row, "gene_base"),
                "gene": getattr(row, "gene"),
                "color_seq_wta": wta_seq,
                "decoded_seq": target_seq,
                "corrected_round_0based": round_idx,
                "corrected_round_1based": round_idx + 1,
                "wta_color": wta_color,
                "target_color": target_color,
                "wta_intensity": wta_intensity,
                "target_intensity": target_intensity,
                "intensity_ratio": (
                    target_intensity / wta_intensity if wta_intensity > 0 else np.nan
                ),
                "wta_prob": wta_prob,
                "target_prob": target_prob,
                "prob_ratio": target_prob / wta_prob if wta_prob > 0 else np.nan,
                "prob_gap": wta_prob - target_prob,
                "score_delta": safe_float(getattr(row, "score_delta")),
                "geomean_prob": safe_float(getattr(row, "geomean_prob")),
                "corrected_round_margin": safe_float(
                    getattr(row, "corrected_round_margin")
                ),
                "mean_total_intensity": safe_float(
                    getattr(row, "mean_total_intensity")
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_h1_tensor_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    finite_score_delta = metrics["score_delta"].replace([np.inf, -np.inf], np.nan)
    work = metrics.assign(finite_score_delta=finite_score_delta)
    return (
        work.groupby("gene_base", dropna=False)
        .agg(
            n=("spot_id", "size"),
            median_intensity_ratio=("intensity_ratio", "median"),
            q25_intensity_ratio=("intensity_ratio", lambda values: values.quantile(0.25)),
            q75_intensity_ratio=("intensity_ratio", lambda values: values.quantile(0.75)),
            frac_intensity_ratio_ge_0p9=(
                "intensity_ratio",
                lambda values: float((values >= 0.90).mean()),
            ),
            frac_intensity_ratio_ge_0p75=(
                "intensity_ratio",
                lambda values: float((values >= 0.75).mean()),
            ),
            frac_intensity_ratio_ge_0p5=(
                "intensity_ratio",
                lambda values: float((values >= 0.50).mean()),
            ),
            median_target_prob=("target_prob", "median"),
            median_wta_prob=("wta_prob", "median"),
            median_prob_gap=("prob_gap", "median"),
            median_geomean_prob=("geomean_prob", "median"),
            median_score_delta_finite=("finite_score_delta", "median"),
            n_score_delta_infinite=("score_delta", lambda values: int(np.isinf(values).sum())),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )


def write_h1_tensor_metrics(
    decoded: pd.DataFrame,
    tensor: np.ndarray,
    genes: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = all_rescued_h1_tensor_metrics(decoded, tensor, genes)
    summary = summarize_h1_tensor_metrics(metrics)
    metrics.to_csv(output_dir / "all_rescued_h1_tensor_metrics.csv", index=False)
    summary.to_csv(output_dir / "all_rescued_h1_tensor_summary.csv", index=False)
    return metrics, summary


def patch_bounds(center: int, radius: int, upper: int) -> tuple[int, int, int]:
    start = max(0, center - radius)
    stop = min(upper, center + radius + 1)
    center_in_patch = center - start
    return start, stop, center_in_patch


def center_window_mean(channel_patch: np.ndarray, cy: int, cx: int, radius: int = 2) -> float:
    y0, y1, _ = patch_bounds(cy, radius, channel_patch.shape[0])
    x0, x1, _ = patch_bounds(cx, radius, channel_patch.shape[1])
    return float(channel_patch[y0:y1, x0:x1].mean())


def patch_metrics(
    patch: np.ndarray,
    *,
    center_y: int,
    center_x: int,
    target_ch: int | None,
    wta_ch: int | None,
) -> dict[str, Any]:
    channel_metrics = []
    for channel_idx in range(patch.shape[2]):
        channel_patch = patch[:, :, channel_idx].astype(np.float64)
        background = float(np.median(channel_patch))
        center_mean = center_window_mean(channel_patch, center_y, center_x)
        channel_metrics.append(
            {
                "channel_idx": channel_idx,
                "peak": float(channel_patch.max()),
                "center_mean": center_mean,
                "background": background,
                "center_to_background": center_mean / (background + 1.0),
                "peak_to_background": float(channel_patch.max()) / (background + 1.0),
            }
        )

    center_values = np.array([metric["center_mean"] for metric in channel_metrics])
    peak_values = np.array([metric["peak"] for metric in channel_metrics])
    center_order = np.argsort(center_values)[::-1]
    peak_order = np.argsort(peak_values)[::-1]

    def channel_value(channel: int | None, key: str) -> float:
        if channel is None:
            return np.nan
        return float(channel_metrics[channel][key])

    def channel_rank(channel: int | None, order: np.ndarray) -> float:
        if channel is None:
            return np.nan
        return float(np.where(order == channel)[0][0] + 1)

    target_center = channel_value(target_ch, "center_mean")
    wta_center = channel_value(wta_ch, "center_mean")
    return {
        "patch_target_center_mean": target_center,
        "patch_wta_center_mean": wta_center,
        "patch_target_peak": channel_value(target_ch, "peak"),
        "patch_wta_peak": channel_value(wta_ch, "peak"),
        "patch_target_center_to_background": channel_value(
            target_ch,
            "center_to_background",
        ),
        "patch_wta_center_to_background": channel_value(
            wta_ch,
            "center_to_background",
        ),
        "patch_target_peak_to_background": channel_value(
            target_ch,
            "peak_to_background",
        ),
        "patch_wta_peak_to_background": channel_value(wta_ch, "peak_to_background"),
        "patch_target_center_rank": channel_rank(target_ch, center_order),
        "patch_target_peak_rank": channel_rank(target_ch, peak_order),
        "patch_target_to_wta_center_ratio": (
            float(target_center / wta_center)
            if np.isfinite(target_center) and wta_center > 0
            else np.nan
        ),
    }


def draw_crosshair(ax: plt.Axes, x: int, y: int, color: str) -> None:
    ax.axhline(y, color=color, linewidth=0.8, alpha=0.75)
    ax.axvline(x, color=color, linewidth=0.8, alpha=0.75)


def patch_scaling(patch: np.ndarray) -> tuple[float, float]:
    values = patch.astype(np.float64).ravel()
    vmin = float(np.percentile(values, 1))
    vmax = float(np.percentile(values, 99.7))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def style_axis(
    ax: plt.Axes,
    *,
    channel_idx: int,
    target_ch: int | None,
    wta_ch: int | None,
) -> None:
    color = "0.55"
    linewidth = 1.0
    if target_ch == channel_idx and wta_ch == channel_idx:
        color = "#2b8a3e"
        linewidth = 2.2
    elif target_ch == channel_idx:
        color = "#2b8a3e"
        linewidth = 2.2
    elif wta_ch == channel_idx:
        color = "#c92a2a"
        linewidth = 2.2
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(linewidth)


def montage_filename(row: pd.Series, suffix: str) -> str:
    category = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["category"]))
    return f"{int(row['example_id']):03d}_{category}_spot{int(row['spot_id'])}_{suffix}.png"


def save_round_montage(
    row: pd.Series,
    *,
    stack_cache: StackCache,
    output_dir: Path,
    patch_radius: int,
) -> tuple[Path, dict[str, Any]]:
    z_idx = int(row["z"])
    y_idx = int(row["y"])
    x_idx = int(row["x"])
    round_idx = int(row["visual_round_0based"])
    target_ch = (
        int(row["visual_target_color"]) - 1
        if str(row["visual_target_color"]).isdigit()
        else None
    )
    wta_ch = (
        int(row["visual_wta_color"]) - 1
        if str(row["visual_wta_color"]).isdigit()
        else None
    )

    plane = stack_cache.get_plane(round_idx, z_idx)
    y0, y1, cy = patch_bounds(y_idx, patch_radius, plane.shape[0])
    x0, x1, cx = patch_bounds(x_idx, patch_radius, plane.shape[1])
    patch = plane[y0:y1, x0:x1, :]
    metrics = patch_metrics(patch, center_y=cy, center_x=cx, target_ch=target_ch, wta_ch=wta_ch)

    vmin, vmax = patch_scaling(patch)
    fig, axes = plt.subplots(1, patch.shape[2], figsize=(3.2 * patch.shape[2], 3.5))
    if patch.shape[2] == 1:
        axes = [axes]
    for channel_idx, ax in enumerate(axes):
        ax.imshow(patch[:, :, channel_idx], cmap="gray", vmin=vmin, vmax=vmax)
        draw_crosshair(ax, cx, cy, "#f08c00")
        role = []
        if channel_idx == target_ch:
            role.append("target")
        if channel_idx == wta_ch:
            role.append("WTA")
        title = f"ch{channel_idx + 1}"
        if role:
            title += f" ({'/'.join(role)})"
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        style_axis(ax, channel_idx=channel_idx, target_ch=target_ch, wta_ch=wta_ch)

    fig.suptitle(
        (
            f"{row['category']} spot {int(row['spot_id'])} "
            f"z/y/x={z_idx}/{y_idx}/{x_idx} round {round_idx + 1}\n"
            f"WTA {row['color_seq_wta']} -> target {row['target_seq']} "
            f"gene={row['target_gene_base']}"
        ),
        fontsize=10,
    )
    fig.tight_layout()
    path = output_dir / "round_montages" / montage_filename(row, "round")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, metrics


def save_all_round_montage(
    row: pd.Series,
    *,
    stack_cache: StackCache,
    output_dir: Path,
    patch_radius: int,
    n_rounds: int,
    n_channels: int,
) -> Path:
    z_idx = int(row["z"])
    y_idx = int(row["y"])
    x_idx = int(row["x"])
    wta_seq = normalize_sequence(row["color_seq_wta"])
    target_seq = normalize_sequence(row["target_seq"])
    gene_label = row.get("target_gene_base") or row.get("gene_base") or row.get("gene")
    if gene_label is None or pd.isna(gene_label) or str(gene_label) == "":
        gene_label = "None"
    decoded_label = target_seq if target_seq else "None"
    call_type = row.get("call_type", "")
    reject_reason = row.get("reject_reason", "")
    reject_suffix = (
        f" reject={reject_reason}"
        if reject_reason is not None and not pd.isna(reject_reason) and str(reject_reason)
        else ""
    )

    fig, axes = plt.subplots(
        n_rounds,
        n_channels,
        figsize=(2.3 * n_channels, 2.1 * n_rounds),
    )
    for round_idx in range(n_rounds):
        plane = stack_cache.get_plane(round_idx, z_idx)
        y0, y1, cy = patch_bounds(y_idx, patch_radius, plane.shape[0])
        x0, x1, cx = patch_bounds(x_idx, patch_radius, plane.shape[1])
        patch = plane[y0:y1, x0:x1, :]
        vmin, vmax = patch_scaling(patch)
        target_ch = (
            int(target_seq[round_idx]) - 1
            if target_seq and target_seq[round_idx].isdigit()
            else None
        )
        wta_ch = (
            int(wta_seq[round_idx]) - 1
            if round_idx < len(wta_seq) and wta_seq[round_idx].isdigit()
            else None
        )
        for channel_idx in range(n_channels):
            ax = axes[round_idx, channel_idx]
            ax.imshow(patch[:, :, channel_idx], cmap="gray", vmin=vmin, vmax=vmax)
            draw_crosshair(ax, cx, cy, "#f08c00")
            ax.set_xticks([])
            ax.set_yticks([])
            if channel_idx == 0:
                ax.set_ylabel(f"r{round_idx + 1}", fontsize=8)
            if round_idx == 0:
                ax.set_title(f"ch{channel_idx + 1}", fontsize=8)
            style_axis(ax, channel_idx=channel_idx, target_ch=target_ch, wta_ch=wta_ch)

    fig.suptitle(
        (
            f"{row['category']} | {call_type}{reject_suffix} | "
            f"gene={gene_label} | spot={int(row['spot_id'])}\n"
            f"WTA={wta_seq} | decoded={decoded_label}"
        ),
        fontsize=10,
    )
    fig.tight_layout()
    path = output_dir / "all_round_montages" / montage_filename(row, "all_rounds")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def summarize_groups(examples: pd.DataFrame) -> pd.DataFrame:
    if examples.empty:
        return pd.DataFrame()
    return (
        examples.groupby(["category", "target_gene_base", "call_group"], dropna=False)
        .agg(
            n=("spot_id", "size"),
            median_score_delta=("score_delta", "median"),
            median_geomean_prob=("geomean_prob", "median"),
            median_target_prob=("visual_target_prob", "median"),
            median_wta_prob=("visual_wta_prob", "median"),
            median_target_to_wta_intensity_ratio=(
                "visual_target_to_wta_intensity_ratio",
                "median",
            ),
            median_patch_target_center_to_background=(
                "patch_target_center_to_background",
                "median",
            ),
            median_patch_wta_center_to_background=(
                "patch_wta_center_to_background",
                "median",
            ),
            fraction_target_center_rank1=(
                "patch_target_center_rank",
                lambda values: float((values == 1).mean()),
            ),
            fraction_target_peak_rank1=(
                "patch_target_peak_rank",
                lambda values: float((values == 1).mean()),
            ),
        )
        .reset_index()
        .sort_values(["category", "target_gene_base", "call_group"])
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        args.result_dir
        / args.dataset
        / "qc"
        / f"{args.fov}_{args.config}_gene_evidence"
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
    tensor_metric_genes = list(dict.fromkeys([*args.genes, *args.control_genes, "Kalrn"]))
    _tensor_metrics, tensor_summary = write_h1_tensor_metrics(
        merged,
        tensor,
        tensor_metric_genes,
        output_dir,
    )
    if args.tensor_metrics_only:
        print(
            f"Saved tensor metrics for {len(tensor_metric_genes)} genes under "
            f"{output_dir}"
        )
        return

    selected = select_examples(
        merged,
        seq_to_gene=seq_to_gene,
        n_channels=n_channels,
        n_rounds=n_rounds,
        target_genes=args.genes,
        control_genes=args.control_genes,
        examples_per_group=args.examples_per_group,
        seed=args.seed,
    )
    annotated_rows = [
        annotate_example(row, tensor=tensor, seq_to_gene=seq_to_gene)
        for _, row in selected.iterrows()
    ]
    examples = pd.DataFrame(annotated_rows)

    stack_cache = StackCache(
        stack_dir=stack_dir_for(args.dataset, args.fov, args.result_dir),
        handles={},
    )
    try:
        round_paths = []
        all_round_paths = []
        metric_rows = []
        for _, row in examples.iterrows():
            round_path, metrics = save_round_montage(
                row,
                stack_cache=stack_cache,
                output_dir=output_dir,
                patch_radius=args.patch_radius,
            )
            record = row.to_dict()
            record.update(metrics)
            record["round_montage_path"] = str(round_path)
            round_paths.append(str(round_path))
            metric_rows.append(record)

        examples = pd.DataFrame(metric_rows)
        for _, row in examples.head(args.all_round_limit).iterrows():
            path = save_all_round_montage(
                row,
                stack_cache=stack_cache,
                output_dir=output_dir,
                patch_radius=args.patch_radius,
                n_rounds=n_rounds,
                n_channels=n_channels,
            )
            all_round_paths.append(str(path))
    finally:
        stack_cache.close()

    examples.to_csv(output_dir / "qc_examples.csv", index=False)
    group_summary = summarize_groups(examples)
    group_summary.to_csv(output_dir / "qc_group_metrics.csv", index=False)

    category_counts = examples["category"].value_counts().sort_index().to_dict()
    summary = {
        "dataset": args.dataset,
        "fov": args.fov,
        "config": args.config,
        "codebook_path": str(codebook_path),
        "split_index": split_index,
        "n_examples": int(len(examples)),
        "category_counts": category_counts,
        "output_dir": str(output_dir),
        "round_montage_count": len(round_paths),
        "all_round_montage_count": len(all_round_paths),
        "target_genes": args.genes,
        "control_genes": args.control_genes,
        "tensor_metric_genes": tensor_metric_genes,
        "tensor_metric_summary_rows": int(len(tensor_summary)),
    }
    (output_dir / "qc_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_safe)
    )

    print(
        f"Saved {len(examples)} QC examples, {len(round_paths)} round montages, "
        f"and {len(all_round_paths)} all-round montages under {output_dir}"
    )


if __name__ == "__main__":
    main()
