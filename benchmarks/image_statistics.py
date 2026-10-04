"""Intensity, puncta and SNR statistics of one channel volume, real or synthetic (W-238).

Development measurement that sets target ranges for the calibrated synthetic presets
(W-241) and the §2.5 rerun (W-239); not D04 calibration and not data qualification.
Definitions are in docs/image-statistics.md. measure_volume takes any integer ZYX
volume that returns YX planes by index: a NumPy array (for example one channel of a
starfinder.synthetic scene round) or a TiffVolume, which reads a TIFF one page at a
time. Histogram statistics accumulate plane by plane over the full volume; puncta are
measured on the central YX crop over all Z planes.

    uv run python ../../benchmarks/image_statistics.py measure <volume.tif>
    uv run python ../../benchmarks/image_statistics.py run --root <data root> --output <run dir>/measurement --pilot
    uv run python ../../benchmarks/image_statistics.py run --root <data root> --output <run dir>/measurement \\
        --pilot-manifest <pilot dir>/manifest.json
    uv run python ../../benchmarks/image_statistics.py attach-time --output <dir> --time-log <file>

run reads only files named *_ch00.tif to *_ch03.tif under <root>/<dataset>/round*/<FOV>/ and
writes nothing under the root.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import ndimage, optimize, stats
import tifffile

SCHEMA = "starfinder.benchmark.image_statistics/1"
#: Histogram percentiles (inverted CDF over integer grey levels).
PERCENTILES = (50, 90, 99, 99.9, 99.99)
DEPTH_PERCENTILE = 99.9
FOOTPRINT_ZYX = (3, 5, 5)
#: Adaptive detection threshold as a fraction of the channel (full-volume) maximum.
ADAPTIVE_FRACTION = 0.2
#: Lenient detection threshold in grey levels.
LENIENT_LEVEL = 10
#: Same-z annulus: the ANNULUS x ANNULUS window minus its central CORE x CORE.
ANNULUS = 21
CORE = 7
CROP_YX = 512
MAX_PUNCTA = 1000
SEED = 0
SELECTIONS = ("adaptive", "lenient")
#: Per-volume truncated lognormal fit: minimum puncta, maximum sigma_log and minimum
#: fitted fraction at or above the truncation for an identified fit.
MIN_FIT_PUNCTA = 20
MAX_FIT_SIGMA = 3.0
MIN_FIT_RETAINED = 0.001

# --- Real-data plan (batch guidance for W-238) --------------------------------------
DATASETS = ("tissue-2D", "cell-culture-3D", "LN", "aging")
#: Register IDs from docs/datasets.md; aging is a catalog version without a benchmark ID.
REGISTER_IDS = {"tissue-2D": "D01", "cell-culture-3D": "D02", "LN": "D03", "aging": None}
CHANNELS = ("ch00", "ch01", "ch02", "ch03")
FOV_POSITIONS = (0, 1, 2, 3)
HELD_OUT_POSITION = 2
TIME_BUDGET_SECONDS = 35 * 60
#: Pre-authorized reductions, applied in order until the projection fits.
REDUCTIONS = (
    dict(name="three_fovs", description="three FOVs per dataset, positions i = 0, 2 and 3 (i = 2 held out)",
         fov_positions=(0, 2, 3)),
    dict(name="smaller_crop", description="a 384x384 crop and at most 500 puncta per selection",
         crop_yx=384, max_puncta=500),
)
PUNCTUM_COLUMNS = ("peak", "background", "clutter_sigma", "pixel_sigma", "amplitude", "snr_clutter", "snr_pixel",
                   "background_fraction", "clutter_pixel_ratio")
LABELS = ("dataset", "fov", "split", "round", "channel")


# --- Volumes ------------------------------------------------------------------------------

class TiffVolume:
    """Read-only lazy ZYX view of a single-series TIFF; volume[z] reads one page."""

    def __init__(self, path):
        self.path = Path(path)
        self._file = tifffile.TiffFile(self.path)
        series = self._file.series[0]
        if len(series.shape) != 3 or len(series.pages) != series.shape[0]:
            self._file.close()
            raise ValueError(f"{self.path}: expected one ZYX series with one page per plane, got {series.shape}")
        self._pages = series.pages
        self.shape = tuple(int(n) for n in series.shape)
        self.dtype = np.dtype(series.dtype)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, z):
        return self._pages[z].asarray()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def synthetic_channel(scene, round_label, channel_label):
    """ZYX view of one channel of a starfinder.synthetic FormedScene round."""
    return scene.rounds[round_label][..., scene.channel_labels.index(channel_label)]


def source_record(path):
    """Path, size and modification time of an input file."""
    info = Path(path).stat()
    return dict(path=str(path), bytes=info.st_size, mtime_ns=info.st_mtime_ns,
                mtime_utc=datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat())


# --- Histogram statistics -----------------------------------------------------------------

def _levels(dtype):
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise TypeError(f"integer uint8 or uint16 volumes are required, got {dtype}")
    return int(np.iinfo(dtype).max) + 1


def quantile_level(counts, percent):
    """Smallest grey level whose cumulative count reaches ceil(percent / 100 * n) (inverted CDF)."""
    counts = np.asarray(counts, dtype=np.int64)
    n = int(counts.sum())
    if n == 0:
        raise ValueError("empty histogram")
    rank = max(1, math.ceil(Fraction(str(percent)) * n / 100))
    return int(np.searchsorted(np.cumsum(counts), rank, side="left"))


def histogram_statistics(counts):
    """Zero and saturated fractions, percentiles, maximum and MAD of a grey-level histogram."""
    counts = np.asarray(counts, dtype=np.int64)
    n = int(counts.sum())
    median = quantile_level(counts, 50)
    deviations = np.bincount(np.abs(np.arange(counts.size) - median), weights=counts, minlength=counts.size)
    result = dict(voxels=n, zero_fraction=float(counts[0] / n), saturated_fraction=float(counts[-1] / n))
    for percent in PERCENTILES:
        result[_percent_key(percent)] = quantile_level(counts, percent)
    result.update(max=int(np.flatnonzero(counts)[-1]), mad=quantile_level(np.rint(deviations).astype(np.int64), 50))
    return result


def _percent_key(percent):
    return "p" + f"{percent:g}".replace(".", "_")


def depth_quarter(z, planes):
    """Depth quarter 0-3 of plane index z in a volume of the given plane count (first to last plane)."""
    return np.minimum(3, (4 * np.asarray(z)) // planes)


def depth_attenuation(profile):
    """Mean per-plane p99.9 of the first depth quarter over that of the last (None if undefined)."""
    profile = np.asarray(profile, dtype=np.float64)
    quarters = depth_quarter(np.arange(profile.size), profile.size)
    first, last = profile[quarters == 0], profile[quarters == 3]
    if first.size == 0 or last.size == 0 or last.mean() <= 0:
        return None
    return float(first.mean() / last.mean())


# --- Puncta -------------------------------------------------------------------------------

def detect_maxima(stack, threshold, *, footprint=FOOTPRINT_ZYX, margin_yx=0):
    """ZYX coordinates of 3D local maxima at or above threshold (and above 0), raster order.

    A voxel is a maximum when it equals the maximum of its footprint (zero outside the
    stack). Each 26-connected plateau of equal maxima counts once, at its first voxel in
    raster order. Maxima within margin_yx of the Y or X edge are dropped.
    """
    stack = np.asarray(stack)
    mask = (stack == ndimage.maximum_filter(stack, size=footprint, mode="constant", cval=0))
    mask &= (stack >= threshold) & (stack > 0)
    if margin_yx:
        mask[:, :margin_yx] = mask[:, -margin_yx:] = False
        mask[:, :, :margin_yx] = mask[:, :, -margin_yx:] = False
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
    coordinates = np.argwhere(mask)
    if count == 0:
        return coordinates
    _, first = np.unique(labels[mask], return_index=True)
    return coordinates[np.sort(first)]


def _annulus_mask():
    ring = np.ones((ANNULUS, ANNULUS), bool)
    lo, hi = (ANNULUS - CORE) // 2, (ANNULUS + CORE) // 2
    ring[lo:hi, lo:hi] = False
    return ring


def punctum_statistics(stack, zyx):
    """Per-punctum peak, local background, clutter and pixel sigma, amplitude and SNRs.

    Over the same-z ANNULUS x ANNULUS window minus its central CORE x CORE: background is
    the median, clutter_sigma the standard deviation, and pixel_sigma the standard
    deviation of horizontal and vertical neighbouring-pixel differences (both pixels in
    the annulus) divided by sqrt(2). amplitude = peak - background; SNRs divide the
    amplitude by each sigma (NaN when the sigma is 0). The window must lie in the stack.
    """
    stack = np.asarray(stack)
    zyx = np.asarray(zyx, dtype=np.intp).reshape(-1, 3)
    half = ANNULUS // 2
    z, y, x = zyx.T
    if len(zyx) and (y.min() < half or x.min() < half or y.max() >= stack.shape[1] - half
                     or x.max() >= stack.shape[2] - half):
        raise ValueError("every punctum's annulus window must lie inside the stack")
    offsets = np.arange(-half, half + 1)
    windows = stack[z[:, None, None], y[:, None, None] + offsets[None, :, None],
                    x[:, None, None] + offsets[None, None, :]].astype(np.float64)
    ring = _annulus_mask()
    annulus = windows[:, ring]
    horizontal = (windows[:, :, 1:] - windows[:, :, :-1])[:, ring[:, 1:] & ring[:, :-1]]
    vertical = (windows[:, 1:, :] - windows[:, :-1, :])[:, ring[1:, :] & ring[:-1, :]]
    peak = stack[z, y, x].astype(np.float64)
    background = np.median(annulus, axis=1)
    clutter = annulus.std(axis=1)
    pixel = np.concatenate([horizontal, vertical], axis=1).std(axis=1) / np.sqrt(2)
    amplitude = peak - background
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(peak=peak, background=background, clutter_sigma=clutter, pixel_sigma=pixel, amplitude=amplitude,
                    snr_clutter=np.where(clutter > 0, amplitude / clutter, np.nan),
                    snr_pixel=np.where(pixel > 0, amplitude / pixel, np.nan),
                    background_fraction=np.where(amplitude > 0, background / amplitude, np.nan),
                    clutter_pixel_ratio=np.where(pixel > 0, clutter / pixel, np.nan))


def central_crop(shape, size):
    """(y0, y1, x0, x1) of the central size x size YX crop (the full extent if smaller)."""
    box = []
    for extent in shape[1:]:
        width = min(size, extent)
        start = (extent - width) // 2
        box += [start, start + width]
    return tuple(box)


def _percentiles(values, name, q=(10, 50, 90)):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {f"{name}_p{p}": (float(np.percentile(values, p)) if values.size else None) for p in q}


@dataclass
class VolumeMeasurement:
    """One volume's record (JSON-ready), its measured puncta and, on request, its crop."""

    record: dict
    puncta: pd.DataFrame
    crop: np.ndarray | None = None


def measure_volume(volume, *, crop_yx=CROP_YX, max_puncta=MAX_PUNCTA, seed=SEED, lenient_level=LENIENT_LEVEL,
                   keep_crop=False):
    """Measure one integer ZYX channel volume (NumPy array or TiffVolume) in one plane-by-plane pass.

    The full volume gives the histogram statistics and the per-plane p99.9 depth profile;
    the central crop_yx crop over all Z gives the puncta. Each selection (adaptive:
    ADAPTIVE_FRACTION x the volume maximum; lenient: lenient_level) keeps maxima whose
    annulus lies in the crop, subsampled to max_puncta with numpy default_rng(seed).
    """
    shape = tuple(int(n) for n in volume.shape)
    if len(shape) != 3:
        raise ValueError(f"expected a ZYX volume, got shape {shape}")
    levels = _levels(volume.dtype)
    y0, y1, x0, x1 = central_crop(shape, crop_yx)
    total = np.zeros(levels, np.int64)
    depth, planes = [], []
    for z in range(shape[0]):
        plane = np.asarray(volume[z])
        if plane.shape != shape[1:]:
            raise ValueError(f"plane {z} has shape {plane.shape}, expected {shape[1:]}")
        counts = np.bincount(plane.ravel(), minlength=levels)
        total += counts
        depth.append(quantile_level(counts, DEPTH_PERCENTILE))
        planes.append(np.array(plane[y0:y1, x0:x1]))
    crop = np.stack(planes)
    record = dict(shape=list(shape), dtype=str(np.dtype(volume.dtype)), crop_yx=[y0, y1, x0, x1],
                  **histogram_statistics(total), depth_p99_9=depth, depth_attenuation=depth_attenuation(depth))
    rng = np.random.default_rng(seed)
    tables = []
    for selection, threshold in zip(SELECTIONS, (ADAPTIVE_FRACTION * record["max"], float(lenient_level))):
        zyx = detect_maxima(crop, threshold, margin_yx=ANNULUS // 2)
        found = len(zyx)
        if found > max_puncta:
            zyx = zyx[np.sort(rng.choice(found, max_puncta, replace=False))]
        table = pd.DataFrame(dict(selection=selection, z=zyx[:, 0], y=zyx[:, 1] + y0, x=zyx[:, 2] + x0,
                                  depth_quarter=depth_quarter(zyx[:, 0], shape[0]), **punctum_statistics(crop, zyx)))
        summary = dict(threshold=float(threshold), n_maxima=found, n_measured=len(table),
                       zero_clutter_fraction=float((table.clutter_sigma == 0).mean()) if len(table) else None,
                       amplitude_by_depth_quarter=[
                           (float(part.median()) if len(part) else None)
                           for part in (table.amplitude[table.depth_quarter == q] for q in range(4))])
        for column in PUNCTUM_COLUMNS:
            summary.update(_percentiles(table[column], column))
        record[selection] = summary
        tables.append(table)
    record["puncta_seed"] = seed
    return VolumeMeasurement(record, pd.concat(tables, ignore_index=True), crop if keep_crop else None)


# --- Summaries ----------------------------------------------------------------------------

def amplitude_breakdown(puncta, by):
    """Median punctum amplitude and count per level of `by` (for example round, channel or depth_quarter)."""
    grouped = puncta.groupby(by, sort=True).amplitude
    return pd.DataFrame(dict(n=grouped.size(), amplitude_p50=grouped.median()))


def max_min_ratio(values):
    """max / min of positive finite values (None when undefined)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or values.min() <= 0:
        return None
    return float(values.max() / values.min())


def truncated_lognormal_fit(values, lower):
    """Maximum-likelihood (mu, sigma_log) of a lognormal left-truncated at `lower`.

    `values` must be at or above `lower`. Returns (None, None) with fewer than
    MIN_FIT_PUNCTA values, a failed optimization, or an unidentified fit: sigma_log
    above MAX_FIT_SIGMA, or less than MIN_FIT_RETAINED of the fitted distribution at
    or above `lower` (the likelihood flattens when only the far tail is observed).
    """
    logs = np.log(np.asarray(values, dtype=np.float64))
    if logs.size < MIN_FIT_PUNCTA or lower <= 0 or np.ptp(logs) == 0:
        return None, None
    cut = math.log(lower)

    def negative_log_likelihood(theta):
        mu, sigma = theta[0], math.exp(theta[1])
        return (logs.size * theta[1] + np.sum((logs - mu) ** 2) / (2 * sigma ** 2)
                + logs.size * stats.norm.logsf((cut - mu) / sigma))

    start = np.array([logs.mean(), math.log(logs.std())])
    fit = optimize.minimize(negative_log_likelihood, start, method="Nelder-Mead",
                            options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
    if not fit.success or not np.all(np.isfinite(fit.x)):
        return None, None
    mu, sigma = float(fit.x[0]), float(math.exp(fit.x[1]))
    if sigma > MAX_FIT_SIGMA or stats.norm.sf((cut - mu) / sigma) < MIN_FIT_RETAINED:
        return None, None
    return mu, sigma


def volume_table(records):
    """One flat row per measured volume."""
    rows = []
    for item in records:
        record = item["record"]
        row = {**item["label"], "path": item["source"]["path"] if item.get("source") else None,
               "seconds": item.get("seconds")}
        row.update({k: v for k, v in record.items() if not isinstance(v, (dict, list))})
        for selection in SELECTIONS:
            row.update({f"{selection}_{k}": v for k, v in record[selection].items() if not isinstance(v, list)})
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_lower(threshold, dtype):
    # Integer peaks: continuity-corrected truncation at the first included grey level minus 0.5.
    return max(0.5, math.ceil(threshold) - 0.5) if dtype.startswith("uint") else threshold


def target_table(volumes, puncta):
    """Per dataset, split (development or held_out) and selection: the calibration target statistics."""
    rows = []
    volume_keys = ["dataset", "fov", "round", "channel"]
    for (dataset, split, selection), group in puncta.groupby(["dataset", "split", "selection"], sort=False):
        volumes_part = volumes[(volumes.dataset == dataset) & (volumes.split == split)]
        row = dict(dataset=dataset, register_id=REGISTER_IDS.get(dataset), split=split, selection=selection,
                   n_volumes=len(volumes_part), n_maxima=int(volumes_part[f"{selection}_n_maxima"].sum()),
                   n_puncta=len(group), threshold_p50=float(volumes_part[f"{selection}_threshold"].median()),
                   threshold_min=float(volumes_part[f"{selection}_threshold"].min()),
                   threshold_max=float(volumes_part[f"{selection}_threshold"].max()))
        for column in PUNCTUM_COLUMNS:
            row.update(_percentiles(group[column], column))
        row["zero_fraction_p50"] = float(volumes_part.zero_fraction.median())
        row["zero_fraction_min"] = float(volumes_part.zero_fraction.min())
        row["zero_fraction_max"] = float(volumes_part.zero_fraction.max())
        row["mad_max"] = float(volumes_part.mad.max())
        channels = amplitude_breakdown(group, "channel")
        rounds = amplitude_breakdown(group, "round")
        quarters = amplitude_breakdown(group, "depth_quarter")
        row["channel_gain_spread"] = max_min_ratio(channels.amplitude_p50)
        row["round_trend"] = max_min_ratio(rounds.amplitude_p50)
        attenuation = volumes_part.depth_attenuation.astype(float)
        attenuation = attenuation[np.isfinite(attenuation)]
        row["depth_attenuation_p50"] = float(attenuation.median()) if len(attenuation) else None
        row["depth_attenuation_min"] = float(attenuation.min()) if len(attenuation) else None
        row["depth_attenuation_max"] = float(attenuation.max()) if len(attenuation) else None
        first, last = quarters.amplitude_p50.reindex(range(4))[[0, 3]]
        row["amplitude_first_last_quarter"] = float(first / last) if last > 0 else None
        positive = group[group.amplitude > 0]
        logs = np.log(positive.amplitude)
        row["sigma_log_amplitude_pooled"] = float(logs.std(ddof=0)) if len(logs) else None
        centred = logs - logs.groupby([positive[k] for k in volume_keys]).transform("mean")
        row["sigma_log_amplitude_within_volume"] = float(centred.std(ddof=0)) if len(centred) else None
        sigmas = []
        thresholds = volumes_part.set_index(volume_keys)[[f"{selection}_threshold", "dtype"]]
        for key, part in group.groupby(volume_keys, sort=False):
            threshold, dtype = thresholds.loc[key]
            sigmas.append(truncated_lognormal_fit(part.peak, _fit_lower(threshold, dtype))[1])
        fitted = np.array([s for s in sigmas if s is not None])
        row["sigma_log_peak_truncated_fit_p50"] = float(np.median(fitted)) if fitted.size else None
        row["sigma_log_peak_truncated_fit_p10"] = float(np.percentile(fitted, 10)) if fitted.size else None
        row["sigma_log_peak_truncated_fit_p90"] = float(np.percentile(fitted, 90)) if fitted.size else None
        row["sigma_log_fits"] = f"{fitted.size}/{len(sigmas)}"
        row["truncation"] = (f"peak >= {ADAPTIVE_FRACTION:g} x volume max (threshold median "
                             f"{row['threshold_p50']:.3g}, range {row['threshold_min']:.3g}-{row['threshold_max']:.3g})"
                             if selection == "adaptive" else f"peak >= {LENIENT_LEVEL} grey levels")
        rows.append(row)
    return pd.DataFrame(rows)


def breakdown_table(puncta):
    """Median amplitude by round, by channel and by depth quarter, per dataset, split and selection."""
    parts = []
    for (dataset, split, selection), group in puncta.groupby(["dataset", "split", "selection"], sort=False):
        for factor in ("round", "channel", "depth_quarter"):
            table = amplitude_breakdown(group, factor).reset_index().rename(columns={factor: "level"})
            table.insert(0, "factor", factor)
            table.insert(0, "selection", selection)
            table.insert(0, "split", split)
            table.insert(0, "dataset", dataset)
            parts.append(table.astype({"level": str}))
    return pd.concat(parts, ignore_index=True)


RANGE_COLUMNS = ("peak_p50", "amplitude_p10", "amplitude_p50", "amplitude_p90", "snr_clutter_p10", "snr_clutter_p50",
                 "snr_clutter_p90", "snr_pixel_p10", "snr_pixel_p50", "snr_pixel_p90", "background_fraction_p50",
                 "clutter_sigma_p50", "pixel_sigma_p50", "clutter_pixel_ratio_p50", "zero_fraction_p50",
                 "channel_gain_spread", "round_trend", "depth_attenuation_p50", "sigma_log_amplitude_within_volume",
                 "sigma_log_peak_truncated_fit_p50")


def target_ranges(targets):
    """Development target range per selection: min and max over datasets of each target statistic."""
    rows = []
    for selection, group in targets[targets.split == "development"].groupby("selection", sort=False):
        for column in RANGE_COLUMNS:
            values = group[column].astype(float)
            rows.append(dict(selection=selection, statistic=column, minimum=float(values.min()),
                             maximum=float(values.max()), datasets=len(values.dropna())))
    return pd.DataFrame(rows)


# --- Real-data plan and run ---------------------------------------------------------------

def fov_positions_indices(n, positions=FOV_POSITIONS):
    """Indices round(i * (n - 1) / 3) for the given positions i."""
    return [round(i * (n - 1) / 3) for i in positions]


def fov_plan(root, datasets=DATASETS, positions=FOV_POSITIONS):
    """Per dataset: the sorted round1 FOV names, the chosen FOVs, their split and the rounds."""
    root = Path(root)
    plan = {}
    for dataset in datasets:
        names = sorted(p.name for p in (root / dataset / "round1").iterdir() if p.is_dir())
        rounds = sorted((p.name for p in (root / dataset).iterdir() if p.is_dir() and re.fullmatch(r"round\d+", p.name)),
                        key=lambda name: int(name[5:]))
        indices = fov_positions_indices(len(names))
        chosen = [dict(position=i, index=indices[i], fov=names[indices[i]],
                       split="held_out" if i == HELD_OUT_POSITION else "development") for i in positions]
        plan[dataset] = dict(register_id=REGISTER_IDS.get(dataset), round1_fov_count=len(names),
                             indices_all_positions=indices, fovs=chosen, rounds=rounds)
    return plan


def volume_jobs(root, plan, rounds=None):
    """(label, path) per chosen FOV, round and channel; missing or ambiguous files are errors."""
    jobs = []
    for dataset, entry in plan.items():
        for fov in entry["fovs"]:
            for round_label in (rounds or entry["rounds"]):
                directory = Path(root) / dataset / round_label / fov["fov"]
                for channel in CHANNELS:
                    hits = sorted(directory.glob(f"*_{channel}.tif"))
                    if len(hits) != 1:
                        raise FileNotFoundError(f"{directory}: expected one *_{channel}.tif, found {len(hits)}")
                    jobs.append((dict(dataset=dataset, fov=fov["fov"], split=fov["split"], round=round_label,
                                      channel=channel), hits[0]))
    return jobs


def project(pilot_manifest, plan, fov_positions):
    """Projected full-run seconds: per dataset, the pilot's mean seconds per volume x planned volumes."""
    pilot = json.loads(Path(pilot_manifest).read_text())
    seconds = pd.DataFrame([dict(dataset=v["dataset"], seconds=v["seconds"]) for v in pilot["volumes"]])
    means = seconds.groupby("dataset").seconds.mean()
    overhead = pilot.get("summary_seconds", 0.0)
    per_dataset = {d: float(means[d] * len(fov_positions) * len(e["rounds"]) * len(CHANNELS)) for d, e in plan.items()}
    return dict(pilot_manifest=str(pilot_manifest), fov_positions=list(fov_positions),
                mean_seconds_per_volume={d: float(means[d]) for d in plan},
                volumes={d: len(fov_positions) * len(e["rounds"]) * len(CHANNELS) for d, e in plan.items()},
                per_dataset_seconds=per_dataset, summary_seconds=float(overhead),
                total_seconds=float(sum(per_dataset.values()) + overhead))


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_revision():
    spec = importlib.util.spec_from_file_location("benchmark_revision", Path(__file__).resolve().parent / "revision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The shared revision record (benchmarks/revision.py) of this checkout.
_revision = _load_revision().revision


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def _csv_bytes(frame):
    return frame.to_csv(index=False).encode()


def _nan_to_none(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    return value


def render_overlay(dataset, crops, path, window=256):
    """One PNG per dataset: per channel, the plane with most adaptive maxima, a central window and its maxima."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(10, 10.8), layout="constrained")
    for axis, (label, crop, threshold) in zip(axes.ravel(), crops):
        adaptive = detect_maxima(crop, threshold, margin_yx=ANNULUS // 2)
        lenient = detect_maxima(crop, LENIENT_LEVEL, margin_yx=ANNULUS // 2)
        z = int(np.bincount(adaptive[:, 0], minlength=crop.shape[0]).argmax()) if len(adaptive) else crop.shape[0] // 2
        h, w = crop.shape[1:]
        y0, x0 = max(0, (h - window) // 2), max(0, (w - window) // 2)
        plane = crop[z, y0:y0 + window, x0:x0 + window]
        top = max(1.0, float(np.percentile(crop, 99.9)))
        axis.imshow(plane, cmap="gray", vmin=0, vmax=top, interpolation="nearest")

        def inside(points):
            points = points[points[:, 0] == z]
            keep = ((points[:, 1] >= y0) & (points[:, 1] < y0 + window) & (points[:, 2] >= x0)
                    & (points[:, 2] < x0 + window))
            return points[keep]
        strong = inside(adaptive)
        adaptive_set = {tuple(p) for p in adaptive}
        weak = np.array([p for p in inside(lenient) if tuple(p) not in adaptive_set]).reshape(-1, 3)
        axis.scatter(weak[:, 2] - x0, weak[:, 1] - y0, s=30, facecolors="none", edgecolors="cyan", linewidths=0.6,
                     label=f"lenient only (>= {LENIENT_LEVEL})")
        axis.scatter(strong[:, 2] - x0, strong[:, 1] - y0, s=60, facecolors="none", edgecolors="red", linewidths=0.9,
                     label=f"adaptive (>= {threshold:.3g})")
        axis.set_title(f"{label['fov']} {label['round']} {label['channel']}, z={z}, crop window "
                       f"y{y0}:{y0 + window} x{x0}:{x0 + window}\ndisplay 0-{top:.0f}; "
                       f"{len(strong)} adaptive, {len(weak)} lenient-only maxima in this plane", fontsize=8)
        axis.legend(fontsize=7, loc="lower right")
        axis.set_axis_off()
    figure.suptitle(f"{dataset}: 3D local maxima (3x5x5) on one plane of the central crop", fontsize=10)
    buffer = io.BytesIO()
    figure.savefig(buffer, dpi=100, format="png")
    plt.close(figure)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())  # one full-file write (network mounts)


def run(root, output, *, pilot=False, pilot_manifest=None, crop_yx=CROP_YX, max_puncta=MAX_PUNCTA, seed=SEED,
        datasets=DATASETS, log=print):
    """Measure the planned real volumes and write records, tables, overlays and a manifest to `output`."""
    started = time.perf_counter()
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("the output directory must not be inside the data root")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} exists and is not empty")
    full_plan = fov_plan(root, datasets)
    positions, reductions, projections = FOV_POSITIONS, [], []
    rounds = None
    if pilot:
        positions, rounds = (0,), ["round1"]
    elif pilot_manifest is not None:
        projection = project(pilot_manifest, full_plan, positions)
        projections.append(dict(stage="base", **projection))
        for reduction in REDUCTIONS:
            if projection["total_seconds"] <= TIME_BUDGET_SECONDS:
                break
            positions = reduction.get("fov_positions", positions)
            crop_yx, max_puncta = reduction.get("crop_yx", crop_yx), reduction.get("max_puncta", max_puncta)
            reductions.append(reduction)
            projection = project(pilot_manifest, full_plan, positions)
            projections.append(dict(stage=f"after {reduction['name']}", **projection,
                                    note="crop reduction projected with the pilot's 512 crop timing (conservative)"
                                    if "crop_yx" in reduction else None))
        if projection["total_seconds"] > TIME_BUDGET_SECONDS:
            raise RuntimeError(f"projected {projection['total_seconds']:.0f} s exceeds the "
                               f"{TIME_BUDGET_SECONDS} s budget after all reductions")
    plan = fov_plan(root, datasets, positions)
    jobs = volume_jobs(root, plan, rounds)
    inputs = [source_record(path) for _, path in jobs]
    log(f"{len(jobs)} volumes; positions {positions}; crop {crop_yx}; max puncta {max_puncta}")
    output.mkdir(parents=True, exist_ok=True)
    records, puncta, overlay_crops = [], [], {}
    for number, (label, path) in enumerate(jobs):
        overlay = (label["fov"] == plan[label["dataset"]]["fovs"][0]["fov"]
                   and label["round"] == (rounds or plan[label["dataset"]]["rounds"])[0])
        tick = time.perf_counter()
        with TiffVolume(path) as volume:
            measured = measure_volume(volume, crop_yx=crop_yx, max_puncta=max_puncta, seed=seed, keep_crop=overlay)
        seconds = time.perf_counter() - tick
        records.append(dict(label=label, source=inputs[number], seconds=round(seconds, 3),
                            record=_nan_to_none(measured.record)))
        table = measured.puncta
        for key in reversed(LABELS):
            table.insert(0, key, label[key])
        puncta.append(table)
        if overlay:
            overlay_crops.setdefault(label["dataset"], []).append(
                (label, measured.crop, measured.record["adaptive"]["threshold"]))
            if len(overlay_crops[label["dataset"]]) == len(CHANNELS):
                render_overlay(label["dataset"], overlay_crops.pop(label["dataset"]),
                               output / "overlays" / f"{label['dataset']}.png")
        log(f"{number + 1}/{len(jobs)} {label['dataset']} {label['fov']} {label['round']} {label['channel']} "
            f"{seconds:.1f}s")
    summary_started = time.perf_counter()
    puncta = pd.concat(puncta, ignore_index=True)
    volumes = volume_table(records)
    targets = target_table(volumes, puncta)
    files = {
        "volumes.jsonl": "".join(json.dumps(r, default=_json_default) + "\n" for r in records).encode(),
        "puncta.csv.gz": gzip.compress(_csv_bytes(puncta), mtime=0),
        "tables/volumes.csv": _csv_bytes(volumes),
        "tables/targets.csv": _csv_bytes(targets),
        "tables/target_ranges.csv": _csv_bytes(target_ranges(targets)),
        "tables/amplitude_breakdown.csv": _csv_bytes(breakdown_table(puncta)),
    }
    for relative, data in files.items():
        (output / relative).parent.mkdir(parents=True, exist_ok=True)
        (output / relative).write_bytes(data)
    summary_seconds = time.perf_counter() - summary_started
    after = [source_record(path) for _, path in jobs]
    changed = [a["path"] for a, b in zip(inputs, after) if (a["bytes"], a["mtime_ns"]) != (b["bytes"], b["mtime_ns"])]
    import matplotlib
    import scipy
    manifest = dict(
        schema=SCHEMA, issue="W-238", scope="pilot" if pilot else "full",
        created_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(), output=str(output),
        code=dict(**_revision(), module="benchmarks/image_statistics.py", module_sha256=_digest(__file__)),
        command=[sys.executable, *sys.argv],
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         pandas=pd.__version__, tifffile=tifffile.__version__, matplotlib=matplotlib.__version__,
                         cpu_affinity=sorted(os.sched_getaffinity(0)),
                         threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                         pythonpath=os.environ.get("PYTHONPATH"),
                         uv_project_environment=os.environ.get("UV_PROJECT_ENVIRONMENT")),
        parameters=dict(percentiles=list(PERCENTILES), depth_percentile=DEPTH_PERCENTILE,
                        footprint_zyx=list(FOOTPRINT_ZYX), adaptive_fraction=ADAPTIVE_FRACTION,
                        lenient_level=LENIENT_LEVEL, annulus=ANNULUS, core=CORE, crop_yx=crop_yx,
                        max_puncta_per_selection=max_puncta, subsample_seed=seed,
                        subsample_rng="numpy.random.default_rng(seed), fresh per volume; adaptive then lenient",
                        min_fit_puncta=MIN_FIT_PUNCTA, max_fit_sigma=MAX_FIT_SIGMA,
                        min_fit_retained=MIN_FIT_RETAINED),
        data_root=str(root), data_root_access="read-only; no file is written, copied, converted or renamed under it",
        channels=list(CHANNELS), excluded_channels=["ch04 (not a sequencing channel)"],
        fov_rule="sorted round1 FOV names; indices round(i * (n - 1) / 3), i = 0..3; i = 2 held out",
        fov_plan=plan, qualification="unqualified: no independent split exists yet (W-92); development targets only",
        time_budget_seconds=TIME_BUDGET_SECONDS, projection=projections, reductions=reductions,
        volumes=[dict(**r["label"], seconds=r["seconds"]) for r in records],
        summary_seconds=round(summary_seconds, 3), compute_seconds=round(time.perf_counter() - started, 3),
        inputs=inputs, inputs_unchanged_after_run=not changed, inputs_changed=changed,
        limitations=[
            "Post-deconvolution uint8 Huygens exports only (CMLE, byte export scaled by one factor per volume).",
            "The uint16 and raw-acquisition intensity scale is not measured and not verified.",
            "Detection keeps local maxima at or above 0.2 x the volume maximum (or 10 grey levels), so the "
            "statistics describe a selection biased towards bright puncta.",
            "Development targets for W-241 and W-239, not D04 calibration; the FOVs are unqualified (W-92).",
            "Depth quarters follow the stored z order; which end is the sample surface is not verified."])
    manifest["files"] = [dict(path=str(p.relative_to(output)), bytes=p.stat().st_size, sha256=_digest(p))
                         for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"]
    manifest["artifact_bytes"] = sum(f["bytes"] for f in manifest["files"])
    (output / "manifest.json").write_text(json.dumps(_nan_to_none(manifest), indent=1, default=_json_default) + "\n")
    return manifest


def attach_time(output, time_log):
    """Add a /usr/bin/time -v record (wall time, maximum RSS) and artifact bytes to the manifest."""
    output = Path(output)
    text = Path(time_log).read_text()
    record = {}
    for line in text.splitlines():
        if ":" in line:
            key, _, value = line.strip().rpartition(": ")
            record[key.strip()] = value.strip()
    wall = record.get("Elapsed (wall clock) time (h:mm:ss or m:ss)")
    seconds = None
    if wall:
        parts = [float(p) for p in wall.split(":")]
        seconds = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rss = int(record["Maximum resident set size (kbytes)"])
    manifest["time_v"] = dict(wall_seconds=seconds, max_rss_kib=rss, exit_status=int(record.get("Exit status", -1)),
                              within_bounds=dict(wall=seconds is not None and seconds <= TIME_BUDGET_SECONDS,
                                                 rss=rss <= 4 * 1024 * 1024),
                              raw=text.splitlines())
    total = sum(p.stat().st_size for p in output.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest["artifact_bytes_total"] = total
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest["time_v"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    single = sub.add_parser("measure", help="measure one ZYX TIFF and print its JSON record")
    single.add_argument("path", type=Path)
    single.add_argument("--crop", type=int, default=CROP_YX)
    single.add_argument("--max-puncta", type=int, default=MAX_PUNCTA)
    single.add_argument("--seed", type=int, default=SEED)
    single.add_argument("--puncta", type=Path, help="also write the measured puncta as CSV (outside Git)")
    runner = sub.add_parser("run", help="measure the planned real datasets")
    runner.add_argument("--root", type=Path, required=True, help="read-only data root")
    runner.add_argument("--output", type=Path, required=True, help="new or empty directory outside Git and the root")
    mode = runner.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="first chosen FOV and round1 of each dataset")
    mode.add_argument("--pilot-manifest", type=Path, help="pilot manifest for the projection and reductions")
    runner.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    timer = sub.add_parser("attach-time", help="add a /usr/bin/time -v record to the manifest")
    timer.add_argument("--output", type=Path, required=True)
    timer.add_argument("--time-log", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "measure":
        with TiffVolume(args.path) as volume:
            measured = measure_volume(volume, crop_yx=args.crop, max_puncta=args.max_puncta, seed=args.seed)
        if args.puncta:
            args.puncta.write_bytes(_csv_bytes(measured.puncta))
        print(json.dumps(dict(source=source_record(args.path), record=_nan_to_none(measured.record)),
                         default=_json_default))
    elif args.command == "run":
        run(args.root, args.output, pilot=args.pilot, pilot_manifest=args.pilot_manifest, datasets=args.datasets,
            log=lambda message: print(message, flush=True))
    else:
        print(json.dumps(attach_time(args.output, args.time_log)["max_rss_kib"]))


if __name__ == "__main__":
    main()
