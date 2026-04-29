#!/usr/bin/env python3
"""Benchmark the lightweight codebook-aware decoder on saved intensity tensors."""

from __future__ import annotations

import argparse
import gc
import json
import re
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from starfinder.barcode import decode_codebook_aware, extract_intensity_tensor, load_codebook
from starfinder.benchmark.validation import compare_genes

BENCHMARK_ROOT = Path("/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark")
POSTCODE_ROOT = BENCHMARK_ROOT / "decoding" / "postcode"
DEFAULT_OUTPUT_DIR = BENCHMARK_ROOT / "decoding" / "codebook_aware" / "results"
AGING_CODEBOOK = Path("/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/aging/genes.csv")
CELL_CULTURE_3D_ROOT = Path(
    "/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/cell-culture-3D"
)
TISSUE_2D_ROOT = Path(
    "/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/tissue-2D"
)
PAD_SUFFIX_RE = re.compile(r"_(STAR|RIBO)_pad_\d+$")

CONFIGS: dict[str, dict[str, Any]] = {
    "wta_exact": {"allow_rescue": False},
    "conservative": {
        "min_corrected_round_margin": 0.10,
        "min_score_delta": 0.50,
        "min_geomean_prob": 0.55,
        "max_correction_penalty": 0.75,
    },
    "balanced": {
        "min_corrected_round_margin": 0.20,
        "min_score_delta": 0.25,
        "min_geomean_prob": 0.45,
        "max_correction_penalty": 1.50,
    },
    "permissive": {
        "min_corrected_round_margin": 0.30,
        "min_score_delta": 0.10,
        "min_geomean_prob": 0.35,
        "max_correction_penalty": 2.50,
    },
}

REAL_DATASETS = {
    "LN": {
        "fovs": ["Position001", "Position002"],
        "fov_arg": "ln_fovs",
        "input_mode": "backend_registered",
        "split_index": None,
    },
    "aging": {
        "fovs": ["Position400"],
        "fov_arg": "aging_fovs",
        "input_mode": "backend_registered",
        "split_index": 4,
    },
    "cell_culture_3D": {
        "fovs": ["Position351", "Position352"],
        "fov_arg": "cell_culture_fovs",
        "input_mode": "raw_e2e",
        "data_root": CELL_CULTURE_3D_ROOT,
        "codebook_path": CELL_CULTURE_3D_ROOT / "genes.csv",
        "sample_id": "cell-culture-3D",
        "output_id": "cell-culture-3D-codebook-aware-input-cache",
        "n_rounds": 6,
        "ref_round": "round1",
        "channel_order": ["ch00", "ch02", "ch01", "ch03"],
        "fov_pattern": "Position%03d",
        "rotate_angle": -90,
        "snr_threshold": 5.0,
        "intensity_estimation": "adaptive",
        "intensity_threshold": 0.2,
        "voxel_size": (1, 2, 2),
        "split_index": None,
    },
    "tissue_2D": {
        "fovs": ["tile_1", "tile_2"],
        "fov_arg": "tissue_2d_fovs",
        "input_mode": "raw_e2e",
        "data_root": TISSUE_2D_ROOT,
        "codebook_path": TISSUE_2D_ROOT / "genes.csv",
        "sample_id": "tissue-2D",
        "output_id": "tissue-2D-codebook-aware-input-cache",
        "n_rounds": 4,
        "ref_round": "round1",
        "channel_order": ["ch00", "ch02", "ch01", "ch03"],
        "fov_pattern": "tile_%d",
        "rotate_angle": -90,
        "snr_threshold": 5.0,
        "intensity_estimation": "adaptive",
        "intensity_threshold": 0.4,
        "voxel_size": (1, 2, 2),
        "split_index": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "synthetic_medium",
            "synthetic_large",
            "LN",
            "aging",
            "cell_culture_3D",
            "tissue_2D",
        ],
        choices=[
            "synthetic_medium",
            "synthetic_large",
            "LN",
            "aging",
            "cell_culture_3D",
            "tissue_2D",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--postcode-root",
        type=Path,
        default=POSTCODE_ROOT,
        help="Existing Postcode benchmark root containing saved tensors and spots.",
    )
    parser.add_argument("--aging-codebook", type=Path, default=AGING_CODEBOOK)
    parser.add_argument("--aging-fovs", nargs="+", default=["Position400"])
    parser.add_argument("--ln-fovs", nargs="+", default=["Position001", "Position002"])
    parser.add_argument(
        "--cell-culture-fovs",
        nargs="+",
        default=["Position351", "Position352"],
    )
    parser.add_argument("--tissue-2d-fovs", nargs="+", default=["tile_1", "tile_2"])
    parser.add_argument("--synthetic-fovs", nargs="+", default=["FOV_001", "FOV_002"])
    parser.add_argument(
        "--reuse-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse cached tensors/spots under the output directory when present.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(CONFIGS),
        choices=list(CONFIGS),
        help="Decoder threshold configs to run.",
    )
    return parser.parse_args()


def strip_aging_pad_suffix(gene_name: str | None) -> str | None:
    if gene_name is None or pd.isna(gene_name):
        return None
    return PAD_SUFFIX_RE.sub("", str(gene_name))


def current_rss_mb() -> float:
    """Return current resident set size in MB on Linux."""
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return np.nan
    pages = int(statm.read_text().split()[1])
    return pages * resource.getpagesize() / (1024 * 1024)


def peak_rss_mb() -> float:
    """Return process peak RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def timed_prep_step(label: str, func) -> tuple[float, float]:
    """Run an input-prep step and return elapsed seconds plus current RSS."""
    t0 = time.perf_counter()
    func()
    elapsed = time.perf_counter() - t0
    rss = current_rss_mb()
    print(f"  {label:<14} {elapsed:8.2f}s  RSS={rss:8.1f}MB")
    return elapsed, rss


def assigned_mask(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str) != "")


def gene_counts(series: pd.Series, *, base_gene: bool) -> pd.Series:
    genes = series.loc[assigned_mask(series)].astype(str)
    if base_gene:
        genes = genes.map(strip_aging_pad_suffix)
    return genes.dropna().value_counts()


def spearman_count_corr(a: pd.Series, b: pd.Series) -> float:
    genes = sorted(set(a.index) | set(b.index))
    if len(genes) < 2:
        return np.nan
    av = np.array([a.get(g, 0) for g in genes], dtype=float)
    bv = np.array([b.get(g, 0) for g in genes], dtype=float)
    if np.all(av == av[0]) or np.all(bv == bv[0]):
        return np.nan
    return float(spearmanr(av, bv).statistic)


def top_jaccard(a: pd.Series, b: pd.Series, n: int = 10) -> float:
    a_top = set(a.sort_values(ascending=False).head(n).index)
    b_top = set(b.sort_values(ascending=False).head(n).index)
    union = a_top | b_top
    return len(a_top & b_top) / len(union) if union else np.nan


def load_saved_inputs(result_dir: Path, fov_id: str) -> tuple[np.ndarray, pd.DataFrame]:
    tensor_path = result_dir / "intensity_tensors" / f"{fov_id}.npy"
    spots_path = result_dir / "spots" / f"{fov_id}.csv"
    if not tensor_path.exists():
        raise FileNotFoundError(tensor_path)
    if not spots_path.exists():
        raise FileNotFoundError(spots_path)
    return np.load(tensor_path), pd.read_csv(spots_path)


def cache_paths(output_dir: Path, dataset: str, fov_id: str) -> tuple[Path, Path]:
    tensor_path = output_dir / dataset / "intensity_tensors" / f"{fov_id}.npy"
    spots_path = output_dir / dataset / "spots" / f"{fov_id}.csv"
    return tensor_path, spots_path


def input_metadata_path(output_dir: Path, dataset: str, fov_id: str) -> Path:
    return output_dir / dataset / "input_metadata" / f"{fov_id}.json"


def cached_raw_input_info(output_dir: Path, dataset: str, fov_id: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "input_source": "cached_raw_e2e_full",
        "input_prep_time_s": 0.0,
    }
    metadata_path = input_metadata_path(output_dir, dataset, fov_id)
    if not metadata_path.exists():
        return info

    metadata = json.loads(metadata_path.read_text())
    original_prep_time = metadata.pop("input_prep_time_s", None)
    original_source = metadata.pop("input_source", None)
    info.update(metadata)
    info["cached_input_source"] = original_source
    info["cached_input_prep_time_s"] = original_prep_time
    return info


def save_cached_inputs(
    tensor: np.ndarray,
    spots: pd.DataFrame,
    output_dir: Path,
    dataset: str,
    fov_id: str,
) -> None:
    tensor_path, spots_path = cache_paths(output_dir, dataset, fov_id)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    spots_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(tensor_path, tensor)
    spots.to_csv(spots_path, index=False)


def load_cached_inputs(
    output_dir: Path,
    dataset: str,
    fov_id: str,
) -> tuple[np.ndarray, pd.DataFrame] | None:
    tensor_path, spots_path = cache_paths(output_dir, dataset, fov_id)
    if not tensor_path.exists() or not spots_path.exists():
        return None
    return np.load(tensor_path), pd.read_csv(spots_path, low_memory=False)


def backend_variant_dir(dataset: str, fov_id: str) -> Path:
    return (
        BENCHMARK_ROOT
        / "e2e_backend_comparison"
        / "results"
        / dataset
        / fov_id
        / "python_global_only"
    )


def raw_real_config(dataset: str) -> dict[str, Any]:
    config = REAL_DATASETS[dataset].copy()
    return {
        "data_root": config["data_root"],
        "codebook_path": config["codebook_path"],
        "sample_id": config["sample_id"],
        "output_id": config["output_id"],
        "n_rounds": config["n_rounds"],
        "ref_round": config["ref_round"],
        "channel_order": config["channel_order"],
        "fov_pattern": config["fov_pattern"],
        "rotate_angle": config["rotate_angle"],
        "snr_threshold": config["snr_threshold"],
        "intensity_estimation": config["intensity_estimation"],
        "intensity_threshold": config["intensity_threshold"],
        "voxel_size": config["voxel_size"],
        "split_index": config.get("split_index"),
    }


def selected_real_fovs(args: argparse.Namespace, dataset: str) -> list[str]:
    fov_arg = REAL_DATASETS[dataset]["fov_arg"]
    return list(getattr(args, fov_arg))


def load_and_register_synthetic_images(
    data_dir: Path,
    fov_id: str,
    n_rounds: int,
) -> dict[str, np.ndarray]:
    from starfinder.io import load_image_stacks
    from starfinder.preprocessing import min_max_normalize
    from starfinder.registration.phase_correlation import register_volume

    images: dict[str, np.ndarray] = {}
    channel_order = ["ch00", "ch01", "ch02", "ch03"]
    for round_idx in range(1, n_rounds + 1):
        round_name = f"round{round_idx}"
        image, _meta = load_image_stacks(
            data_dir / fov_id / round_name,
            channel_order=channel_order,
            convert_uint8=True,
        )
        images[round_name] = min_max_normalize(image, snr_threshold=5.0)

    ref_merged = np.sum(images["round1"], axis=-1, dtype=np.uint16)
    for round_idx in range(2, n_rounds + 1):
        round_name = f"round{round_idx}"
        mov_merged = np.sum(images[round_name], axis=-1, dtype=np.uint16)
        registered, _shifts = register_volume(images[round_name], ref_merged, mov_merged)
        images[round_name] = registered

    return images


def detect_synthetic_spots(images: dict[str, np.ndarray]) -> pd.DataFrame:
    from starfinder.spotfinding import find_spots_3d

    spots = find_spots_3d(
        images["round1"],
        intensity_estimation="noise",
        intensity_threshold=5.0,
        min_distance=1,
    )
    spots = spots.reset_index(drop=True)
    spots.insert(0, "spot_id", np.arange(len(spots), dtype=int))
    return spots


def prepare_synthetic_inputs(
    *,
    dataset: str,
    data_dir: Path,
    fov_id: str,
    ground_truth: dict[str, Any],
    output_dir: Path,
    reuse_inputs: bool,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    if reuse_inputs:
        cached = load_cached_inputs(output_dir, dataset, fov_id)
        if cached is not None:
            tensor, spots = cached
            return tensor, spots, {"input_source": "cached", "input_prep_time_s": 0.0}

    t0 = time.perf_counter()
    images = load_and_register_synthetic_images(data_dir, fov_id, ground_truth["n_rounds"])
    spots = detect_synthetic_spots(images)
    round_order = [f"round{i}" for i in range(1, ground_truth["n_rounds"] + 1)]
    tensor = extract_intensity_tensor(images, spots, round_order)
    prep_time = time.perf_counter() - t0
    save_cached_inputs(tensor, spots, output_dir, dataset, fov_id)

    return tensor, spots, {
        "input_source": "generated_synthetic",
        "input_prep_time_s": round(prep_time, 3),
    }


def load_registered_real_images(variant_dir: Path, n_rounds: int) -> dict[str, np.ndarray]:
    from starfinder.io import load_multipage_tiff

    stack_dir = variant_dir / "registered_final"
    if not stack_dir.exists():
        stack_dir = variant_dir / "registered_global"

    images: dict[str, np.ndarray] = {}
    for round_idx in range(1, n_rounds + 1):
        round_name = f"round{round_idx}"
        images[round_name] = load_multipage_tiff(
            stack_dir / f"{round_name}.tif",
            convert_uint8=False,
        )
    return images


def prepare_real_backend_inputs(
    *,
    dataset: str,
    fov_id: str,
    output_dir: Path,
    reuse_inputs: bool,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    variant_dir = backend_variant_dir(dataset, fov_id)
    metadata = json.loads((variant_dir / "run_metadata.json").read_text())
    config = metadata["dataset_config"]

    if reuse_inputs:
        cached = load_cached_inputs(output_dir, dataset, fov_id)
        if cached is not None:
            tensor, spots = cached
            return (
                tensor,
                spots,
                config,
                {
                    "input_source": "cached_backend_registered_full",
                    "input_prep_time_s": 0.0,
                },
            )

    t0 = time.perf_counter()
    spots = pd.read_csv(variant_dir / "signal" / f"{fov_id}_allSpots_zero_based.csv")
    spots = spots.reset_index(drop=True)
    if "spot_id" not in spots:
        spots.insert(0, "spot_id", np.arange(len(spots), dtype=int))

    images = load_registered_real_images(variant_dir, int(config["n_rounds"]))
    round_order = [f"round{i}" for i in range(1, int(config["n_rounds"]) + 1)]
    voxel_size = tuple(int(v) for v in config["voxel_size"])
    tensor = extract_intensity_tensor(images, spots, round_order, voxel_size=voxel_size)
    prep_time = time.perf_counter() - t0
    save_cached_inputs(tensor, spots, output_dir, dataset, fov_id)

    return tensor, spots, config, {
        "input_source": "backend_registered_full",
        "input_prep_time_s": round(prep_time, 3),
    }


def prepare_real_raw_inputs(
    *,
    dataset: str,
    fov_id: str,
    output_dir: Path,
    reuse_inputs: bool,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    config = raw_real_config(dataset)

    if reuse_inputs:
        cached = load_cached_inputs(output_dir, dataset, fov_id)
        if cached is not None:
            tensor, spots = cached
            return (
                tensor,
                spots,
                config,
                cached_raw_input_info(output_dir, dataset, fov_id),
            )

    from starfinder.dataset import STARMapDataset
    from starfinder.dataset.types import LayerState

    t0 = time.perf_counter()
    print(f"\nPreparing raw-E2E inputs for {dataset} {fov_id}")
    ds = STARMapDataset(
        input_root=config["data_root"],
        output_root=output_dir / dataset / "raw_pipeline_cache",
        dataset_id="sample-dataset",
        sample_id=config["sample_id"],
        output_id=config["output_id"],
        layers=LayerState(
            seq=[f"round{i}" for i in range(1, int(config["n_rounds"]) + 1)],
            ref=config["ref_round"],
        ),
        channel_order=config["channel_order"],
        fov_pattern=config["fov_pattern"],
    )
    ds.load_codebook(config["codebook_path"], split_index=config.get("split_index"))
    fov = ds.fov(fov_id)

    step_info: dict[str, float] = {}
    elapsed, rss = timed_prep_step("load", fov.load_raw_images)
    step_info["input_time_load_s"] = round(elapsed, 3)
    step_info["input_rss_after_load_mb"] = round(rss, 1)
    elapsed, rss = timed_prep_step(
        "rotate", lambda: fov.rotate(angle=float(config["rotate_angle"]))
    )
    step_info["input_time_rotate_s"] = round(elapsed, 3)
    step_info["input_rss_after_rotate_mb"] = round(rss, 1)
    elapsed, rss = timed_prep_step(
        "enhance",
        lambda: fov.enhance_contrast(snr_threshold=float(config["snr_threshold"])),
    )
    step_info["input_time_enhance_s"] = round(elapsed, 3)
    step_info["input_rss_after_enhance_mb"] = round(rss, 1)
    elapsed, rss = timed_prep_step("registration", fov.global_registration)
    step_info["input_time_registration_s"] = round(elapsed, 3)
    step_info["input_rss_after_registration_mb"] = round(rss, 1)
    elapsed, rss = timed_prep_step(
        "spot_finding",
        lambda: fov.spot_finding(
            intensity_estimation=config["intensity_estimation"],
            intensity_threshold=float(config["intensity_threshold"]),
        ),
    )
    step_info["input_time_spot_finding_s"] = round(elapsed, 3)
    step_info["input_rss_after_spot_finding_mb"] = round(rss, 1)

    if fov.all_spots is None:
        raise ValueError(f"No spots detected for {dataset} {fov_id}")
    spots = fov.all_spots.reset_index(drop=True).copy()
    if "spot_id" not in spots:
        spots.insert(0, "spot_id", np.arange(len(spots), dtype=int))

    round_order = [f"round{i}" for i in range(1, int(config["n_rounds"]) + 1)]
    tensor_t0 = time.perf_counter()
    tensor = extract_intensity_tensor(
        fov.images,
        spots,
        round_order,
        voxel_size=tuple(int(v) for v in config["voxel_size"]),
    )
    tensor_elapsed = time.perf_counter() - tensor_t0
    tensor_rss = current_rss_mb()
    print(f"  {'tensor_extract':<14} {tensor_elapsed:8.2f}s  RSS={tensor_rss:8.1f}MB")
    step_info["input_time_tensor_extract_s"] = round(tensor_elapsed, 3)
    step_info["input_rss_after_tensor_extract_mb"] = round(tensor_rss, 1)

    save_cached_inputs(tensor, spots, output_dir, dataset, fov_id)
    prep_time = time.perf_counter() - t0
    metadata_path = input_metadata_path(output_dir, dataset, fov_id)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": dataset,
        "fov_id": fov_id,
        "input_source": "raw_e2e_full",
        "input_prep_time_s": round(prep_time, 3),
        "n_input_spots": len(spots),
        "tensor_shape": list(tensor.shape),
        "global_shifts": {
            round_name: list(shift)
            for round_name, shift in fov.global_shifts.items()
        },
        "dataset_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in config.items()
        },
        **step_info,
        "input_prep_rss_after_mb": round(current_rss_mb(), 1),
        "input_prep_peak_rss_mb": round(peak_rss_mb(), 1),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return tensor, spots, config, metadata


def prepare_real_inputs(
    *,
    dataset: str,
    fov_id: str,
    output_dir: Path,
    reuse_inputs: bool,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    mode = REAL_DATASETS[dataset]["input_mode"]
    if mode == "backend_registered":
        return prepare_real_backend_inputs(
            dataset=dataset,
            fov_id=fov_id,
            output_dir=output_dir,
            reuse_inputs=reuse_inputs,
        )
    if mode == "raw_e2e":
        return prepare_real_raw_inputs(
            dataset=dataset,
            fov_id=fov_id,
            output_dir=output_dir,
            reuse_inputs=reuse_inputs,
        )
    raise ValueError(f"Unsupported real input mode for {dataset}: {mode}")


def decoded_eval_frame(
    spots: pd.DataFrame,
    decoded: pd.DataFrame,
    *,
    use_wta: bool,
) -> pd.DataFrame:
    merged = spots.merge(decoded, on="spot_id", how="left")
    if use_wta:
        gene = merged["gene_wta"]
        color_seq = merged["color_seq_wta"]
    else:
        gene = merged["gene"]
        color_seq = merged["decoded_seq"]
    return pd.DataFrame(
        {
            "z": merged["z"],
            "y": merged["y"],
            "x": merged["x"],
            "gene": gene,
            "color_seq": color_seq,
        }
    )


def synthetic_metrics(
    spots: pd.DataFrame,
    decoded: pd.DataFrame,
    ground_truth: dict[str, Any],
    fov_id: str,
) -> dict[str, Any]:
    wta_eval = decoded_eval_frame(spots, decoded, use_wta=True)
    cba_eval = decoded_eval_frame(spots, decoded, use_wta=False)
    wta_result = compare_genes(wta_eval, ground_truth, fov_id)
    cba_result = compare_genes(cba_eval, ground_truth, fov_id)
    n_gt = len(ground_truth["fovs"][fov_id]["spots"])

    rescued_ids = set(
        decoded.loc[decoded["call_type"].astype(str).str.startswith("rescued"), "spot_id"]
    )
    rescued_eval = cba_eval.loc[spots["spot_id"].isin(rescued_ids)]
    if len(rescued_eval):
        rescued_result = compare_genes(rescued_eval, ground_truth, fov_id)
        rescued_accuracy = rescued_result["gene_accuracy"]
        wrong_rescue_rate = (
            1.0 - rescued_accuracy if rescued_result["n_matched"] > 0 else np.nan
        )
        rescued_matched = rescued_result["n_matched"]
    else:
        rescued_accuracy = np.nan
        wrong_rescue_rate = np.nan
        rescued_matched = 0

    return {
        "wta_gene_accuracy": wta_result["gene_accuracy"],
        "cba_gene_accuracy": cba_result["gene_accuracy"],
        "wta_color_seq_accuracy": wta_result["color_seq_accuracy"],
        "cba_color_seq_accuracy": cba_result["color_seq_accuracy"],
        "wta_recall": wta_result["correct_genes"] / n_gt if n_gt else 0.0,
        "cba_recall": cba_result["correct_genes"] / n_gt if n_gt else 0.0,
        "rescued_matched": rescued_matched,
        "rescued_gene_accuracy": rescued_accuracy,
        "wrong_rescue_rate": wrong_rescue_rate,
    }


def real_metrics(decoded: pd.DataFrame, *, n_rounds: int) -> dict[str, Any]:
    rescue_mask = decoded["call_type"].astype(str).str.startswith("rescued")
    wta_base = gene_counts(decoded["gene_wta"], base_gene=True)
    cba_base = gene_counts(decoded["gene"], base_gene=True)
    rescued_base = gene_counts(decoded.loc[rescue_mask, "gene"], base_gene=True)

    n_rescued = int(rescue_mask.sum())
    top_fraction = (
        float(rescued_base.iloc[0] / n_rescued)
        if n_rescued and len(rescued_base)
        else 0.0
    )
    top_gene = str(rescued_base.index[0]) if len(rescued_base) else ""

    correction_counts = {f"correction_round_{idx}": 0 for idx in range(n_rounds)}
    for value in decoded.loc[rescue_mask, "corrected_rounds"].dropna().astype(str):
        for token in value.split(","):
            if token == "":
                continue
            key = f"correction_round_{int(token)}"
            if key in correction_counts:
                correction_counts[key] += 1

    return {
        "base_gene_distribution_corr": spearman_count_corr(wta_base, cba_base),
        "top10_gene_jaccard": top_jaccard(wta_base, cba_base),
        "rescued_top_gene_fraction": top_fraction,
        "rescued_top_gene": top_gene,
        **correction_counts,
    }


def base_row(
    *,
    dataset: str,
    fov_id: str,
    config_name: str,
    config: dict[str, Any],
    decoded: pd.DataFrame,
    elapsed_s: float,
    rss_before_mb: float,
    rss_after_mb: float,
    peak_rss_after_mb: float,
    n_genes_codebook: int,
    input_info: dict[str, Any],
) -> dict[str, Any]:
    n_input = len(decoded)
    exact_mask = decoded["call_type"] == "exact"
    rescue_mask = decoded["call_type"].astype(str).str.startswith("rescued")
    assigned = assigned_mask(decoded["gene"])
    wta_assigned = assigned_mask(decoded["gene_wta"])

    call_counts = decoded["call_type"].value_counts().to_dict()
    reject_counts = decoded.loc[~assigned, "reject_reason"].value_counts().to_dict()

    return {
        "dataset": dataset,
        "fov_id": fov_id,
        "config_name": config_name,
        "n_input_spots": n_input,
        "n_genes_codebook": n_genes_codebook,
        "wta_n_assigned": int(wta_assigned.sum()),
        "wta_match_rate": float(wta_assigned.mean()) if n_input else 0.0,
        "cba_n_assigned": int(assigned.sum()),
        "cba_match_rate": float(assigned.mean()) if n_input else 0.0,
        "n_exact": int(exact_mask.sum()),
        "n_rescued": int(rescue_mask.sum()),
        "rescue_rate": float(rescue_mask.mean()) if n_input else 0.0,
        "exact_agreement_rate": 1.0 if int(wta_assigned.sum()) else np.nan,
        "time_s": round(elapsed_s, 3),
        "rss_before_mb": round(rss_before_mb, 1),
        "rss_after_mb": round(rss_after_mb, 1),
        "rss_delta_mb": round(rss_after_mb - rss_before_mb, 1),
        "peak_rss_after_mb": round(peak_rss_after_mb, 1),
        **input_info,
        "call_type_counts": json.dumps(call_counts, sort_keys=True),
        "reject_reason_counts": json.dumps(reject_counts, sort_keys=True),
        **{f"param_{key}": value for key, value in config.items()},
    }


def run_dataset_fov(
    *,
    dataset: str,
    fov_id: str,
    tensor: np.ndarray,
    spots: pd.DataFrame,
    seq_to_gene: dict[str, str],
    output_dir: Path,
    configs: list[str],
    input_info: dict[str, Any] | None = None,
    ground_truth: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    n_rounds = tensor.shape[2]
    input_info = input_info or {}

    for config_name in configs:
        config = CONFIGS[config_name]
        gc.collect()
        rss_before = current_rss_mb()
        t0 = time.perf_counter()
        decoded = decode_codebook_aware(
            tensor,
            seq_to_gene,
            spot_ids=spots["spot_id"].to_numpy(),
            **config,
        )
        elapsed = time.perf_counter() - t0
        rss_after = current_rss_mb()
        peak_after = peak_rss_mb()

        config_dir = output_dir / dataset / "codebook_aware" / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        decoded_path = config_dir / f"{fov_id}.csv"
        decoded.to_csv(decoded_path, index=False)

        row = base_row(
            dataset=dataset,
            fov_id=fov_id,
            config_name=config_name,
            config=config,
            decoded=decoded,
            elapsed_s=elapsed,
            rss_before_mb=rss_before,
            rss_after_mb=rss_after,
            peak_rss_after_mb=peak_after,
            n_genes_codebook=len(seq_to_gene),
            input_info=input_info,
        )
        if ground_truth is not None:
            row.update(synthetic_metrics(spots, decoded, ground_truth, fov_id))
        else:
            row.update(real_metrics(decoded, n_rounds=n_rounds))
        rows.append(row)

        print(
            f"{dataset} {fov_id} {config_name}: "
            f"assigned={row['cba_n_assigned']} rescued={row['n_rescued']} "
            f"match={row['cba_match_rate']:.4f} time={elapsed:.2f}s "
            f"rss={row['rss_after_mb']:.1f}MB peak={row['peak_rss_after_mb']:.1f}MB"
        )

    return rows


def save_dataset_results(output_dir: Path, dataset: str, rows: list[dict[str, Any]]) -> None:
    dataset_dir = output_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dataset_dir / "comparison.csv", index=False)

    numeric = pd.DataFrame(rows).select_dtypes(include=[np.number])
    summary = {
        "dataset": dataset,
        "n_rows": len(rows),
        "metrics_mean": numeric.mean(numeric_only=True).to_dict(),
        "rows": rows,
    }
    (dataset_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_safe)
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value):
        return None
    return value


def run_synthetic_medium(args: argparse.Namespace) -> list[dict[str, Any]]:
    data_dir = args.postcode_root / "data" / "synthetic_medium"
    result_dir = args.postcode_root / "results" / "synthetic_medium"
    ground_truth = json.loads((data_dir / "ground_truth.json").read_text())
    _gene_to_seq, seq_to_gene = load_codebook(data_dir / "codebook.csv")

    rows = []
    for fov_id in args.synthetic_fovs:
        tensor, spots = load_saved_inputs(result_dir, fov_id)
        input_info = {"input_source": "postcode_saved_tensor", "input_prep_time_s": 0.0}
        rows.extend(
            run_dataset_fov(
                dataset="synthetic_medium",
                fov_id=fov_id,
                tensor=tensor,
                spots=spots,
                seq_to_gene=seq_to_gene,
                output_dir=args.output_dir,
                configs=args.configs,
                input_info=input_info,
                ground_truth=ground_truth,
            )
        )
    save_dataset_results(args.output_dir, "synthetic_medium", rows)
    return rows


def run_synthetic_preset(args: argparse.Namespace, preset: str) -> list[dict[str, Any]]:
    dataset = f"synthetic_{preset}"
    data_dir = BENCHMARK_ROOT / "e2e" / "data" / preset
    ground_truth = json.loads((data_dir / "ground_truth.json").read_text())
    _gene_to_seq, seq_to_gene = load_codebook(data_dir / "codebook.csv")

    rows = []
    for fov_id in args.synthetic_fovs:
        tensor, spots, input_info = prepare_synthetic_inputs(
            dataset=dataset,
            data_dir=data_dir,
            fov_id=fov_id,
            ground_truth=ground_truth,
            output_dir=args.output_dir,
            reuse_inputs=args.reuse_inputs,
        )
        rows.extend(
            run_dataset_fov(
                dataset=dataset,
                fov_id=fov_id,
                tensor=tensor,
                spots=spots,
                seq_to_gene=seq_to_gene,
                output_dir=args.output_dir,
                configs=args.configs,
                input_info=input_info,
                ground_truth=ground_truth,
            )
        )
    save_dataset_results(args.output_dir, dataset, rows)
    return rows


def run_real_dataset(args: argparse.Namespace, dataset: str) -> list[dict[str, Any]]:
    rows = []
    fovs = selected_real_fovs(args, dataset)

    seq_to_gene: dict[str, str] | None = None
    for fov_id in fovs:
        tensor, spots, config, input_info = prepare_real_inputs(
            dataset=dataset,
            fov_id=fov_id,
            output_dir=args.output_dir,
            reuse_inputs=args.reuse_inputs,
        )
        if seq_to_gene is None:
            split_index = config.get("split_index")
            _gene_to_seq, seq_to_gene = load_codebook(
                config["codebook_path"],
                split_index=split_index,
            )
        rows.extend(
            run_dataset_fov(
                dataset=dataset,
                fov_id=fov_id,
                tensor=tensor,
                spots=spots,
                seq_to_gene=seq_to_gene,
                output_dir=args.output_dir,
                configs=args.configs,
                input_info=input_info,
                ground_truth=None,
            )
        )
        del tensor, spots
        gc.collect()
    save_dataset_results(args.output_dir, dataset, rows)
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    if "synthetic_medium" in args.datasets:
        all_rows.extend(run_synthetic_medium(args))
    if "synthetic_large" in args.datasets:
        all_rows.extend(run_synthetic_preset(args, "large"))
    if "LN" in args.datasets:
        all_rows.extend(run_real_dataset(args, "LN"))
    if "aging" in args.datasets:
        all_rows.extend(run_real_dataset(args, "aging"))
    if "cell_culture_3D" in args.datasets:
        all_rows.extend(run_real_dataset(args, "cell_culture_3D"))
    if "tissue_2D" in args.datasets:
        all_rows.extend(run_real_dataset(args, "tissue_2D"))

    pd.DataFrame(all_rows).to_csv(args.output_dir / "comparison_all.csv", index=False)
    print(f"\nSaved benchmark results under {args.output_dir}")


if __name__ == "__main__":
    main()
