"""Integer histogram summaries, their merge across FOVs and inverted-CDF percentiles.

Sample-level statistics come from per-FOV histograms taken at a fitted step's
input and summed elementwise; see the preprocessing algorithms page.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path

import numpy as np

from starfinder.image import _validate_image

_DTYPES = ("uint8", "uint16")


def _labels(values, what):
    values = tuple(values)
    if not values or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
        raise ValueError(f"{what} must be unique nonempty strings")
    return values


def _steps_record(entries):
    """JSON-normalized list of {"step": name, "config": {...}} preceding-step records."""
    try:
        entries = json.loads(json.dumps(list(entries), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"summarized_after must be JSON-serializable: {error}") from None
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"step", "config"} or not isinstance(entry["step"], str) \
                or not isinstance(entry["config"], dict):
            raise ValueError('summarized_after entries must be {"step": name, "config": {...}} mappings')
    return entries


@dataclass(frozen=True, eq=False)
class HistogramSummary:
    """Per-channel integer histograms of one or more FOVs at a fitted step's input.

    counts has shape (rounds, channels, bins) with one bin per representable
    value of dtype (uint8 or uint16): bin v counts the voxels equal to v.
    summarized_after lists the preceding recipe steps as
    {"step": name, "config": {...}} records. fovs_used are the FOVs whose
    counts were summed; fovs_excluded were deliberately left out of a merge.
    """
    counts: np.ndarray
    dtype: str
    round_names: tuple[str, ...]
    channel_labels: tuple[str, ...]
    summarized_after: tuple = ()
    fovs_used: tuple[str, ...] = ()
    fovs_excluded: tuple[str, ...] = ()

    def __post_init__(self):
        if self.dtype not in _DTYPES:
            raise ValueError("histogram summaries support uint8 and uint16 only")
        object.__setattr__(self, "round_names", _labels(self.round_names, "round_names"))
        object.__setattr__(self, "channel_labels", _labels(self.channel_labels, "channel_labels"))
        object.__setattr__(self, "fovs_used", _labels(self.fovs_used, "fovs_used"))
        excluded = tuple(self.fovs_excluded)
        if excluded:
            _labels(excluded, "fovs_excluded")
        if set(excluded) & set(self.fovs_used):
            raise ValueError("a FOV cannot be both used and excluded")
        object.__setattr__(self, "fovs_excluded", excluded)
        object.__setattr__(self, "summarized_after", tuple(_steps_record(self.summarized_after)))
        counts = np.asarray(self.counts)
        shape = (len(self.round_names), len(self.channel_labels), np.iinfo(self.dtype).max + 1)
        if counts.shape != shape:
            raise ValueError(f"counts must have shape {shape} (rounds, channels, bins), not {counts.shape}")
        if counts.dtype.kind not in "ui" or (counts < 0).any():
            raise ValueError("counts must be nonnegative integers")
        object.__setattr__(self, "counts", counts.astype(np.int64))


def summarize_histograms(images: Mapping[str, np.ndarray], *, channel_labels: Sequence[str], fov_id: str,
                         summarized_after: Sequence[Mapping] = ()) -> HistogramSummary:
    """Record the per-channel histograms of one FOV's rounds.

    images maps round names, in order, to finite ZYXC (or single-channel ZYX)
    uint8 or uint16 volumes of one dtype, taken at the fitted step's input:
    after the preceding recipe steps (see summary_stage). Each histogram is
    numpy.bincount(values, minlength=dtype_max + 1), with all bins. Inputs are
    not modified.

    Raises
    ------
    ValueError
        No rounds, float or other integer dtypes, mixed dtypes, or a channel
        count that differs from channel_labels.
    """
    if not isinstance(images, Mapping) or not images:
        raise ValueError("images must map at least one round name to a volume")
    channel_labels = _labels(channel_labels, "channel_labels")
    dtypes = set()
    counts = []
    for name, volume in images.items():
        volume = _validate_image(volume)
        dtypes.add(volume.dtype.name)
        if volume.dtype.name not in _DTYPES:
            raise ValueError(f"round {name!r}: histogram summaries support uint8 and uint16 only, not {volume.dtype}")
        channels = volume[..., None] if volume.ndim == 3 else volume
        if channels.shape[-1] != len(channel_labels):
            raise ValueError(f"round {name!r} has {channels.shape[-1]} channels, but {len(channel_labels)} labels were given")
        bins = np.iinfo(volume.dtype).max + 1
        counts.append([np.bincount(channels[..., c].ravel(), minlength=bins) for c in range(channels.shape[-1])])
    if len(dtypes) != 1:
        raise ValueError(f"all rounds must share one dtype, not {sorted(dtypes)}")
    return HistogramSummary(np.asarray(counts, dtype=np.int64), dtypes.pop(), tuple(images), channel_labels,
                            tuple(summarized_after), (fov_id,))


def merge_histograms(summaries: Sequence[HistogramSummary], *, exclude: Sequence[str] = ()) -> HistogramSummary:
    """Sum the histograms of the selected FOVs elementwise.

    Every summary must share dtype, round names, channel labels (hence the bin
    layout) and summarized_after. Summaries of the FOVs named in exclude (for
    example near-empty tissue-edge fields) are left out of the sum and recorded
    in fovs_excluded; every other summary is used.

    Raises
    ------
    ValueError
        No summaries, incompatible summaries, a FOV that appears twice, an
        exclude name that matches no summary or only part of a merged one, or
        nothing left to merge.
    """
    summaries = list(summaries)
    if not summaries or any(not isinstance(s, HistogramSummary) for s in summaries):
        raise ValueError("merge_histograms requires at least one HistogramSummary")
    first = summaries[0]
    for other in summaries[1:]:
        for field in ("dtype", "round_names", "channel_labels", "summarized_after"):
            if getattr(other, field) != getattr(first, field):
                raise ValueError(f"summaries differ in {field}: {getattr(first, field)!r} and {getattr(other, field)!r}")
    fovs = [fov for s in summaries for fov in s.fovs_used + s.fovs_excluded]
    if len(set(fovs)) != len(fovs):
        raise ValueError("a FOV appears in more than one summary")
    exclude = set(exclude)
    unknown = exclude - {fov for s in summaries for fov in s.fovs_used}
    if unknown:
        raise ValueError(f"excluded FOVs {sorted(unknown)} match no summary")
    used, excluded = [], []
    for summary in summaries:
        dropped = exclude & set(summary.fovs_used)
        if dropped and dropped != set(summary.fovs_used):
            raise ValueError(f"exclude names only part of the merged FOVs {list(summary.fovs_used)}")
        (excluded if dropped else used).append(summary)
    if not used:
        raise ValueError("every FOV was excluded")
    counts = np.sum([s.counts for s in used], axis=0, dtype=np.int64)
    return HistogramSummary(counts, first.dtype, first.round_names, first.channel_labels, first.summarized_after,
                            tuple(fov for s in used for fov in s.fovs_used),
                            tuple(fov for s in used for fov in s.fovs_excluded)
                            + tuple(fov for s in excluded for fov in s.fovs_used + s.fovs_excluded))


def _rank(n, p):
    # numpy.percentile(method="inverted_cdf") in its own float64 arithmetic:
    # sorted index ceil(n * p / 100 - 1), clipped to [0, n - 1].
    index = n * np.true_divide(p, 100) - 1
    previous = np.floor(index)
    rank = int(previous if index - previous == 0 else previous + 1)
    return min(max(rank, 0), n - 1)


def histogram_percentile(counts: np.ndarray, p: float) -> int:
    """Inverted-CDF percentile of the values counted by a histogram.

    counts[v] is the number of voxels equal to v. The result is the smallest
    value whose cumulative count reaches p/100 of the total (for p = 0, the
    smallest value with a nonzero count). It equals
    numpy.percentile(values, p, method="inverted_cdf") exactly, including
    numpy's float64 rounding of p/100 times the voxel count.

    Raises
    ------
    ValueError
        counts is not a nonempty 1D array of nonnegative integers with a
        nonzero total, or p is not finite in [0, 100].
    """
    counts = np.asarray(counts)
    if counts.ndim != 1 or counts.size == 0 or counts.dtype.kind not in "ui" or (counts < 0).any():
        raise ValueError("counts must be a nonempty 1D array of nonnegative integers")
    if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)) or not math.isfinite(p) \
            or not 0 <= p <= 100:
        raise ValueError("p must be finite in [0, 100]")
    cumulative = np.cumsum(counts, dtype=np.int64)
    total = int(cumulative[-1])
    if total == 0:
        raise ValueError("counts must have a nonzero total")
    return int(np.searchsorted(cumulative, _rank(total, p) + 1, side="left"))


def write_histograms(summary: HistogramSummary, path: Path | str) -> Path:
    """Save a summary as .npz: counts (rounds, channels, bins), labels, FOVs and preceding steps.

    The file is written in one full-file write and read back by read_histograms.
    """
    if not isinstance(summary, HistogramSummary):
        raise TypeError("write_histograms requires a HistogramSummary")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, counts=summary.counts, dtype=np.str_(summary.dtype),
                        round_names=np.asarray(summary.round_names, dtype=str),
                        channel_labels=np.asarray(summary.channel_labels, dtype=str),
                        summarized_after=np.str_(json.dumps(list(summary.summarized_after))),
                        fovs_used=np.asarray(summary.fovs_used, dtype=str),
                        fovs_excluded=np.asarray(summary.fovs_excluded, dtype=str))
    path = Path(path)
    path.write_bytes(buffer.getvalue())
    return path


def read_histograms(path: Path | str) -> HistogramSummary:
    """Load a summary written by write_histograms, revalidating it."""
    with np.load(Path(path), allow_pickle=False) as data:
        return HistogramSummary(data["counts"], str(data["dtype"]), tuple(data["round_names"].tolist()),
                                tuple(data["channel_labels"].tolist()), tuple(json.loads(str(data["summarized_after"]))),
                                tuple(data["fovs_used"].tolist()), tuple(data["fovs_excluded"].tolist()))
