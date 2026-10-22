"""The appearance bank of the real-appearance cellular scene (docs/synthetic-real-appearance.md).

A bank is a set of entries cut from source images with their method labels, with background patches, stored as
one ``.npz`` file outside the repository and identified by its SHA-256 (ST1), or built in memory from scenes of the
accepted closed-form generator (CF1). An entry is a nucleus, or in the culture a cell with its largest nucleus label
and both stains (EN1). The build keeps a label by the four selection rules (SEL1), stores per entry the signed
distance ``D_e = EDT(not M) − EDT(M)`` of its mask ("Terms"), its excess inside the mask and its own fall-off
outside (OUT1), relative to its source image's plateau (BR1), and refuses a source outside BR4's guard. Its record
lists every input with its SHA-256 (or the preset and seeds), the rules and their counts, the plateaus and a digest
of the arrays, by which a rebuild is compared because a zip archive's bytes carry a time.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import scipy
from scipy import ndimage as ndi
import skimage
from skimage.filters import threshold_otsu

from starfinder._records import file_sha256

from ._sample import _cast, _streams

#: Version of the bank file layout and its build rules; any change to an array, a rule or the record bumps it.
APPEARANCE_BANK_VERSION = "1"
#: The environment variable naming the directory of the working copies of the stored banks (ST1).
BANK_DIRECTORY_VARIABLE = "STARFINDER_APPEARANCE_BANKS"

_HEX = re.compile(r"[0-9a-f]{64}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# The selection counts of a source image, in the order of the rules ("Selection of entries", W-513's columns).
_COUNTS = ("labels", "left_out_border", "left_out_border_z_alone", "left_out_size_after_border",
           "left_out_nucleus_after_size", "left_out_squashed_after_rest", "kept")
_DEFAULTS = dict(
    entry="nucleus", channels=None, spacing_zyx=None, keep_z=None, margin_um=(1.5, 1.5, 1.5), guard=1.5, seed=0,
    selection=dict(border_z=None, size_percentiles=(5.0, 95.0), min_nucleus_voxels=50, rim_share_max=None),
    fall_off=dict(length_um=0.6, blur_um=0.25, smooth_um=0.3, weight_um=0.6, bin_um=0.1, min_bin_voxels=10,
                  min_bins=3),
    patches=dict(shape_zyx=(10, 24, 24), overlap_zyx=(2, 12, 12), per_origin=60, max_foreground=0.10,
                 label_distance_um=1.0, level=0.7, max_tries=20_000, fill_sigma_zyx=(1.0, 2.0, 2.0)))
# Build settings of the closed-form banks by context, after the prototype's per-context values (proto.py, round 10;
# W-513 scripts/measure.py): margins, the culture's whole-stack boxes, patch sizes and overlaps. A shape of 0 in Z is
# every plane of the source. The closed-form banks cut 20 patches per Z origin (the prototype 60 or 300).
_CLOSED_FORM_BUILD = {
    "dense3d": dict(entry="nucleus", keep_z=False, margin_um=(1.5, 1.5, 1.5),
                    patches=dict(shape_zyx=(10, 24, 24), overlap_zyx=(2, 12, 12), per_origin=20)),
    "tissue2d": dict(entry="nucleus", keep_z=False, margin_um=(0.0, 1.5, 1.5),
                     patches=dict(shape_zyx=(1, 24, 24), overlap_zyx=(0, 12, 12), per_origin=20)),
    "culture": dict(entry="cell", keep_z=True, margin_um=(0.0, 1.5, 1.5),
                    patches=dict(shape_zyx=(0, 32, 32), overlap_zyx=(0, 10, 10), per_origin=20)),
}


@dataclass(frozen=True)
class BankReference:
    """Which appearance bank a scene uses (ST1, CF1 of docs/synthetic-real-appearance.md).

    A stored bank is named by ``name`` and ``sha256``: the file ``<name>.npz`` in ``directory``,
    or with ``directory=None`` in the directory that the environment variable
    ``STARFINDER_APPEARANCE_BANKS`` names; ``load_appearance_bank`` refuses a file whose SHA-256
    differs. A closed-form bank is named by ``preset`` (a ``CONTEXT_PRESETS`` name) and ``n``,
    the scenes rendered for it (``closed_form_bank``). Give one of the two. Instances are not
    hashable.
    """

    __hash__ = None

    name: str | None = None
    sha256: str | None = None
    preset: str | None = None
    n: int = 4
    directory: str | None = None

    def __post_init__(self):
        from ._context_presets import CONTEXT_PRESETS
        if self.preset is None:
            if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
                raise ValueError(f"name: a stored bank needs a plain file name without suffix; got {self.name!r}")
            if not isinstance(self.sha256, str) or not _HEX.fullmatch(self.sha256):
                raise ValueError(f"sha256: a stored bank needs a lowercase hexadecimal SHA-256; got {self.sha256!r}")
        else:
            if self.name is not None or self.sha256 is not None:
                raise ValueError("a closed-form bank (preset) takes no name and no sha256")
            if self.preset not in CONTEXT_PRESETS:
                raise ValueError(f"preset: unknown context preset {self.preset!r}; choose from "
                                 f"{list(CONTEXT_PRESETS)}")
            if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 2:
                raise ValueError(f"n: a closed-form bank renders at least 2 scenes (one held out); got {self.n!r}")
        if self.directory is not None and not isinstance(self.directory, (str, os.PathLike)):
            raise ValueError(f"directory must be a path or None; got {self.directory!r}")


class AppearanceBank:
    """An appearance bank in memory: its arrays (read-only), entry and patch tables and build record.

    ``entries`` has one row per entry (``entry_id``, ``source_image``, ``source_label``,
    ``plateau``, the box origin and shape in source voxels, the spacing, ``volume``, ``voxels``,
    ``rim_share``, ``nucleus_voxels`` and per channel the local background and rim level over
    the plateau). ``distance``, ``excess`` and ``nucleus_distance`` return an entry's arrays on
    its box; ``patch`` a background patch (channels, Z, Y, X), relative to its source plateau.
    ``sha256`` is the file's (None for a closed-form bank).
    """

    def __init__(self, arrays, record, *, sha256=None):
        self.arrays = {}
        for key, value in arrays.items():
            value = np.asarray(value)
            value.setflags(write=False)
            self.arrays[key] = value
        self.record = record
        self.sha256 = sha256
        self.name = record["bank"]
        self.digest = record["digest"]
        self.channels = tuple(str(c) for c in self.arrays["channels"])
        a = self.arrays
        table = dict(entry_id=a["entry_id"].astype(object), source_image=a["source_image"].astype(object),
                     source_label=a["source_label"], plateau=a["plateau"],
                     **{f"box_origin_{x}": a["box_origin"][:, i] for i, x in enumerate("zyx")},
                     **{f"shape_{x}": a["box_shape"][:, i] for i, x in enumerate("zyx")},
                     **{f"spacing_{x}": a["spacing"][:, i] for i, x in enumerate("zyx")},
                     volume=a["volume"], voxels=a["voxels"], rim_share=a["rim_share"],
                     nucleus_voxels=a["nucleus_voxels"])
        for c, channel in enumerate(self.channels):
            table[f"local_background_{channel}"] = a["local_background"][:, c]
            table[f"rim_level_{channel}"] = a["rim_level"][:, c]
        self.entries = pd.DataFrame(table)
        self.patches = pd.DataFrame(dict(
            patch_id=np.arange(len(a["patch_image"]), dtype=np.int64), source_image=a["patch_image"].astype(object),
            **{f"origin_{x}": a["patch_origin"][:, i] for i, x in enumerate("zyx")},
            foreground=a["patch_foreground"]))
        self._index = {str(e): i for i, e in enumerate(a["entry_id"])}

    def __len__(self):
        return len(self._index)

    def entry(self, entry_id) -> int:
        """The row of ``entry_id`` in ``entries``; ValueError naming the entry when the bank does not hold it."""
        try:
            return self._index[str(entry_id)]
        except KeyError:
            raise ValueError(f"bank {self.name}: no entry_id {entry_id!r}") from None

    def _flat(self, key, entry):
        i = self.entry(entry) if isinstance(entry, str) else int(entry)
        start = int(self.arrays["offset"][i])
        shape = tuple(int(v) for v in self.arrays["box_shape"][i])
        return self.arrays[key][..., start:start + int(np.prod(shape))].reshape(
            self.arrays[key].shape[:-1] + shape)

    def distance(self, entry) -> np.ndarray:
        """The entry's signed distance ``D_e`` (µm, negative inside its mask) on its box."""
        return self._flat("distance", entry)

    def excess(self, entry, channel=None) -> np.ndarray:
        """The entry's excess over its plateau on its box: every channel (C, Z, Y, X), or one channel by name."""
        values = self._flat("excess", entry)
        return values if channel is None else values[self.channels.index(channel)]

    def nucleus_distance(self, entry) -> np.ndarray:
        """A cell entry's nucleus signed distance on its box; ValueError for a bank of nucleus entries."""
        if self.record["entry"] != "cell":
            raise ValueError(f"bank {self.name}: nucleus entries carry no separate nucleus mask")
        return self._flat("nucleus_distance", entry)

    def patch(self, index) -> np.ndarray:
        """Background patch ``index`` (channels, Z, Y, X), relative to its source plateau."""
        return self.arrays["patch"][int(index)]


# ------------------------------------------------------------------------------------------------ the rules


def _struct(shape):
    """The face-connected structure, without Z neighbours on a single plane."""
    structure = ndi.generate_binary_structure(3, 1)
    if shape[0] == 1:
        structure[0] = structure[2] = False
    return structure


def _sigmas(sigma_um, spacing, shape):
    return [sigma_um / s if n > 1 else 0.0 for s, n in zip(spacing, shape)]


def _signed_distance(mask, spacing):
    """``D_e = EDT(not M) − EDT(M)`` (µm) on the mask's grid ("Terms")."""
    return (ndi.distance_transform_edt(~mask, sampling=spacing)
            - ndi.distance_transform_edt(mask, sampling=spacing))


def _plateau(labels, stain):
    """The median over the labels of each label's median stain; NaN without a label."""
    ids = np.unique(labels[labels > 0])
    if not len(ids):
        return math.nan
    return float(np.median(np.asarray(ndi.median(stain, labels, index=ids), dtype=np.float64)))


def _visible_median(stain):
    """The median stain over the visible stain: smoothed by a Gaussian of one voxel, above its Otsu threshold."""
    smooth = ndi.gaussian_filter(stain, [1.0 if n > 1 else 0.0 for n in stain.shape])
    visible = smooth > threshold_otsu(smooth)
    return float(np.median(stain[visible])) if visible.any() else math.nan


def _boxes(labels, margin, keep_z):
    objects = ndi.find_objects(labels)
    shape = labels.shape
    boxes = {}
    for i, sl in enumerate(objects):
        if sl is None:
            continue
        box = tuple(slice(max(s.start - m, 0), min(s.stop + m, n)) for s, m, n in zip(sl, margin, shape))
        if keep_z:
            box = (slice(0, shape[0]),) + box[1:]
        boxes[i + 1] = (sl, box)
    return boxes


def _select(source, cfg):
    """Which rule leaves out each label, in the order of "Selection of entries" (SEL1), with the rim shares and the
    culture cells' nuclei; returns (the per-label table, the counts, the boxes)."""
    labels = source["labels"]
    shape = labels.shape
    flat = shape[0] == 1
    keep_z = cfg["keep_z"]
    margin = np.ceil(np.asarray(cfg["margin_um"], dtype=np.float64) / source["spacing"]).astype(np.int64)
    if flat or keep_z:
        margin[0] = 0
    boxes = _boxes(labels, margin, keep_z)
    sizes = np.bincount(labels.ravel())
    rows = []
    for label, (sl, box) in boxes.items():
        border_yx = any(s.start == 0 or s.stop == n for s, n in zip(sl[1:], shape[1:]))
        border_z = (not flat) and (sl[0].start == 0 or sl[0].stop == shape[0])
        rows.append(dict(label=label, voxels=int(sizes[label]), border_yx=border_yx, border_z=border_z))
    table = pd.DataFrame(rows, columns=["label", "voxels", "border_yx", "border_z"])
    use_z = cfg["selection"]["border_z"]
    table["rule_border"] = table.border_yx | (table.border_z & bool(use_z) & (not flat))
    passing = table.voxels[~table.rule_border].to_numpy()
    lo, hi = (np.percentile(passing, cfg["selection"]["size_percentiles"]) if len(passing)
              else (math.nan, math.nan))
    table["rule_size"] = ~table.rule_border & ~((table.voxels >= lo) & (table.voxels <= hi))
    structure = _struct(shape)
    nucleus = source.get("nucleus_labels")
    shares, nucleus_ids, nucleus_voxels = [], [], []
    for row in table.itertuples():
        if row.rule_border or row.rule_size:
            shares.append(math.nan), nucleus_ids.append(0), nucleus_voxels.append(0)
            continue
        box = boxes[row.label][1]
        own = labels[box] == row.label
        other = (labels[box] > 0) & ~own
        rim = own & ~ndi.binary_erosion(own, structure=structure)
        touching = rim & ndi.binary_dilation(other, structure=structure)
        shares.append(float(touching.sum() / max(int(rim.sum()), 1)))
        if nucleus is None:
            nucleus_ids.append(0), nucleus_voxels.append(0)
            continue
        inside = nucleus[box][own & (nucleus[box] > 0)]
        if not inside.size:
            nucleus_ids.append(0), nucleus_voxels.append(0)
            continue
        counts = np.bincount(inside)
        nucleus_ids.append(int(counts.argmax())), nucleus_voxels.append(int(counts.max()))
    table["rim_share"] = shares
    table["nucleus_label"] = nucleus_ids
    table["nucleus_voxels"] = nucleus_voxels
    rest = ~table.rule_border & ~table.rule_size
    small = table.nucleus_voxels < cfg["selection"]["min_nucleus_voxels"]
    table["rule_nucleus"] = rest & (nucleus is not None) & small
    table["rule_squashed"] = rest & ~table.rule_nucleus & (table.rim_share > cfg["selection"]["rim_share_max"])
    table["kept"] = rest & ~table.rule_nucleus & ~table.rule_squashed
    counts = dict(labels=len(table), left_out_border=int(table.rule_border.sum()),
                  left_out_border_z_alone=int(table.border_z.sum()) if (use_z and not flat) else 0,
                  left_out_size_after_border=int(table.rule_size.sum()),
                  left_out_nucleus_after_size=int(table.rule_nucleus.sum()),
                  left_out_squashed_after_rest=int(table.rule_squashed.sum()), kept=int(table.kept.sum()),
                  size_p5_voxels=float(lo) if np.isfinite(lo) else None,
                  size_p95_voxels=float(hi) if np.isfinite(hi) else None)
    return table, counts, boxes


def _fall_off(X, own_mask, other, d, r, spacing, cfg):
    """OUT1 for one channel on an entry's box: the excess ``X`` inside the mask, its own capped fall-off outside
    ("What an entry carries", steps 1 to 4), zero beyond ``λ`` + four deviations of the outside blur."""
    shape = X.shape
    length, blur = cfg["length_um"], cfg["blur_um"]
    # 1. The surroundings smoothed over smooth_um, capped by the linear fall-off from the rim level over λ.
    kept = np.minimum(ndi.gaussian_filter(X, _sigmas(cfg["smooth_um"], spacing, shape)),
                      r * np.clip(1.0 - d / length, 0.0, None))
    # 2. The weight v: nearer to the entry than to any other source label, smoothed over weight_um, 0 on others.
    if other.any():
        nearer = (d <= ndi.distance_transform_edt(~other, sampling=spacing)).astype(np.float64)
    else:
        nearer = np.ones(shape, dtype=np.float64)
    v = ndi.gaussian_filter(nearer, _sigmas(cfg["weight_um"], spacing, shape))
    v[other] = 0.0
    # 3. own: the per-distance median of the capped values on the entry's side (v > 0.5), made non-increasing.
    bins = int(round(length / cfg["bin_um"]))
    edges = cfg["bin_um"] * np.arange(bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    near = ~own_mask & (v > 0.5)
    profile = np.full(bins, np.nan)
    for q in range(bins):
        chosen = near & (d > edges[q]) & (d <= edges[q + 1])
        if chosen.sum() >= cfg["min_bin_voxels"]:
            profile[q] = np.median(kept[chosen])
    finite = np.isfinite(profile)
    if finite.sum() >= cfg["min_bins"]:
        profile = np.minimum.accumulate(np.interp(centres, centres[finite], profile[finite]))
        own = np.interp(d, np.concatenate([[0.0], centres, [length]]),
                        np.concatenate([[min(r, profile[0])], profile, [0.0]]))
    else:
        own = r * np.clip(1.0 - d / length, 0.0, None)
    # 4. Outside the mask the combination blurred over blur_um; inside the mask the excess, not blurred.
    outside = kept * v + own * (1.0 - v)
    blurred = ndi.gaussian_filter(np.where(own_mask, X, outside), _sigmas(blur, spacing, shape))
    result = np.where(own_mask, X, blurred)
    result[d > length + 4 * blur] = 0.0
    return result


def _entry(source, label, box, nucleus_label, cfg):
    """The arrays of one kept label: D_e, the excess per channel and (culture) the nucleus D_e, not yet over the
    plateau."""
    spacing = source["spacing"]
    L = source["labels"][box]
    own = L == label
    other = (L > 0) & ~own
    free = ~own & ~other
    rim = own & ~ndi.binary_erosion(own, structure=_struct(own.shape))
    d = ndi.distance_transform_edt(~own, sampling=spacing)
    excess, backgrounds, rims = [], [], []
    for stain in source["stains"]:
        values = stain[box]
        b = float(np.percentile(values[free], 20)) if free.sum() > 20 else 0.0
        X = np.clip(values - b, 0.0, None)
        r = float(np.median(X[rim]))
        excess.append(_fall_off(X, own, other, d, r, spacing, cfg["fall_off"]))
        backgrounds.append(b)
        rims.append(r)
    out = dict(distance=d - ndi.distance_transform_edt(own, sampling=spacing), excess=excess,
               local_background=backgrounds, rim_level=rims, voxels=int(own.sum()))
    if source.get("nucleus_labels") is not None:
        out["nucleus_distance"] = _signed_distance((source["nucleus_labels"][box] == nucleus_label) & own, spacing)
    return out


def _patch_origins(n, size, step):
    origins = list(range(0, max(n - size, 0) + 1, max(step, 1)))
    if origins[-1] != n - size:
        origins.append(n - size)
    return origins


def _patches(source, plateau, cfg, stream):
    """Background patches of a source image (BG1's patch rule): at most ``max_foreground`` of the patch within
    ``label_distance_um`` of a label or above ``level`` of the plateau; holes filled by the patch's local mean plus
    residuals drawn from its own background; levels equalized per Z origin; relative to the plateau."""
    p = cfg["patches"]
    stains = source["stains"]
    shape = stains[0].shape
    size = np.array([shape[0] if p["shape_zyx"][0] == 0 else p["shape_zyx"][0], *p["shape_zyx"][1:]],
                    dtype=np.int64)
    if (size > np.array(shape)).any():
        return [], [], []
    labelled = source["labels"] > 0
    if source.get("nucleus_labels") is not None:
        labelled |= source["nucleus_labels"] > 0
    foreground = ndi.distance_transform_edt(~labelled, sampling=source["spacing"]) < p["label_distance_um"]
    for stain in stains:
        foreground |= stain > p["level"] * plateau
    sigma = [s if n > 1 else 0.0 for s, n in zip(p["fill_sigma_zyx"], size)]
    step_z = int(size[0] - p["overlap_zyx"][0]) if p["shape_zyx"][0] else int(size[0])
    patches, origins, shares = [], [], []
    for z0 in _patch_origins(shape[0], int(size[0]), step_z):
        rng = stream("context.appearance.bank.patch", f"{source['image']}/z-{z0}")
        got = []
        for _ in range(p["max_tries"]):
            if len(got) >= p["per_origin"]:
                break
            y0 = int(rng.integers(0, shape[1] - size[1] + 1))
            x0 = int(rng.integers(0, shape[2] - size[2] + 1))
            box = (slice(z0, z0 + size[0]), slice(y0, y0 + size[1]), slice(x0, x0 + size[2]))
            holes = foreground[box]
            share = float(holes.mean())
            if share > p["max_foreground"]:
                continue
            channels = []
            for stain in stains:
                patch = stain[box].copy()
                if holes.any():
                    keep = (~holes).astype(np.float64)
                    low = (ndi.gaussian_filter(patch * keep, sigma)
                           / np.maximum(ndi.gaussian_filter(keep, sigma), 1e-6))
                    residuals = (patch - low)[~holes]
                    patch[holes] = low[holes] + rng.choice(residuals, size=int(holes.sum()))
                channels.append(patch)
            got.append((np.stack(channels), (z0, y0, x0), share))
        if got:
            level = np.mean([[g[0][c].mean() for c in range(len(stains))] for g in got], axis=0)
            for values, origin, share in got:
                values = np.stack([values[c] - values[c].mean() + level[c] for c in range(len(stains))])
                patches.append(values / plateau)
                origins.append(origin)
                shares.append(share)
    return patches, origins, shares


# ------------------------------------------------------------------------------------------------ configuration


def _merged(defaults, given, name):
    unknown = sorted(set(given) - set(defaults))
    if unknown:
        raise ValueError(f"{name}: unknown keys {unknown}")
    return {k: given.get(k, v) for k, v in defaults.items()}


def _configuration(given):
    """The build settings with defaults; ValueError for an unknown key or an invalid value."""
    top = {k: v for k, v in given.items() if k not in ("bank", "sources", "held_out")}
    cfg = _merged(_DEFAULTS, top, "configuration")
    for part in ("selection", "fall_off", "patches"):
        cfg[part] = _merged(_DEFAULTS[part], dict(top.get(part) or {}), part)
    if cfg["entry"] not in ("nucleus", "cell"):
        raise ValueError(f"entry must be 'nucleus' or 'cell'; got {cfg['entry']!r}")
    cell = cfg["entry"] == "cell"
    cfg["keep_z"] = cell if cfg["keep_z"] is None else bool(cfg["keep_z"])
    selection = cfg["selection"]
    selection["border_z"] = (not cfg["keep_z"]) if selection["border_z"] is None else bool(selection["border_z"])
    if selection["rim_share_max"] is None:
        selection["rim_share_max"] = 0.35 if cell else 0.25
    if not isinstance(cfg["channels"], (list, tuple)) or not cfg["channels"]:
        raise ValueError("channels: name the stains of the entries, the nuclear stain first")
    cfg["channels"] = [str(c) for c in cfg["channels"]]
    spacing = np.asarray(cfg["spacing_zyx"], dtype=np.float64) if cfg["spacing_zyx"] is not None else None
    if spacing is None or spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or (spacing <= 0).any():
        raise ValueError(f"spacing_zyx must be three positive µm; got {cfg['spacing_zyx']!r}")
    cfg["spacing_zyx"] = spacing.tolist()
    if not cfg["guard"] > 1:
        raise ValueError(f"guard must be a factor above 1; got {cfg['guard']!r}")
    for key in ("length_um", "blur_um", "smooth_um", "weight_um", "bin_um"):
        if not cfg["fall_off"][key] > 0:
            raise ValueError(f"fall_off.{key} must be positive; got {cfg['fall_off'][key]!r}")
    cfg["margin_um"] = [float(v) for v in cfg["margin_um"]]
    selection["size_percentiles"] = [float(v) for v in selection["size_percentiles"]]
    for key in ("shape_zyx", "overlap_zyx", "fill_sigma_zyx"):
        cfg["patches"][key] = list(cfg["patches"][key])
    return cfg


# ------------------------------------------------------------------------------------------------ the build


def _digest(arrays):
    """SHA-256 over the arrays by sorted name: name, dtype, shape and C-order bytes."""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(f"{key}\0{value.dtype.str}\0{value.shape}\0".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _guard(sources, plateaus, visible, factor):
    """BR4: a source whose plateau lies outside ``factor`` of the bank's median plateau, or whose plateau and median
    over the visible stain disagree by more than ``factor``, is refused; returns the bank's median plateau."""
    for source, p in zip(sources, plateaus):
        if not (p > 0):
            raise ValueError(f"source {source['image']}: its plateau {p} is not positive (no label or no stain)")
    median = float(np.median(plateaus))
    refused = []
    for source, p, v in zip(sources, plateaus, visible):
        if max(p / median, median / p) > factor:
            refused.append(f"source {source['image']}: its plateau {p:g} lies outside a factor {factor:g} of the "
                           f"bank's median plateau {median:g}")
        if not (v > 0) or max(v / p, p / v) > factor:
            refused.append(f"source {source['image']}: its plateau {p:g} and its median over the visible stain "
                           f"{v:g} disagree by more than a factor {factor:g}")
    if refused:
        raise ValueError("BR4 refuses the build: " + "; ".join(refused))
    return median


def _build(name, sources, held_out, cfg, inputs):
    """The build step on loaded sources: (arrays, record without the file's SHA-256)."""
    images = [s["image"] for s in sources]
    if len(set(images)) != len(images):
        raise ValueError(f"sources: duplicate source image IDs {images}")
    held = {h["image"] for h in held_out}
    clash = sorted(held & set(images))
    if clash:
        raise ValueError(f"held-out images {clash} are among the sources; a held-out image is never a source")
    plateaus, visible = [], []
    for source in sources:
        reference = source["nucleus_labels"] if source.get("nucleus_labels") is not None else source["labels"]
        plateaus.append(_plateau(reference, source["stains"][0]))
        visible.append(_visible_median(source["stains"][0]))
    median = _guard(sources, plateaus, visible, cfg["guard"])
    # The patch draws depend on the seed, the source image and the Z origin only, so the same sources give the same
    # bank whatever its name.
    stream = _streams("bank", int(cfg["seed"]), "appearance-bank", None)
    channels = len(cfg["channels"])
    rows, flats, nucleus_flats, excess_flats = [], [], [], [[] for _ in range(channels)]
    patches, patch_rows = [], []
    image_rows = []
    offset = 0
    for source, plateau, vis in zip(sources, plateaus, visible):
        table, counts, boxes = _select(source, cfg)
        spacing = source["spacing"]
        voxel = float(np.prod(spacing if source["labels"].shape[0] > 1 else spacing[1:]))
        for row in table[table.kept].itertuples():
            box = boxes[row.label][1]
            arrays = _entry(source, row.label, box, row.nucleus_label, cfg)
            shape = tuple(s.stop - s.start for s in box)
            # Each entry's arrays are cast to their float32 storage at once, which holds the build's memory to one
            # float32 copy of the entries (the same values as casting the concatenation).
            flats.append(arrays["distance"].ravel().astype(np.float32))
            for c in range(channels):
                excess_flats[c].append((arrays["excess"][c] / plateau).ravel().astype(np.float32))
            if "nucleus_distance" in arrays:
                nucleus_flats.append(arrays["nucleus_distance"].ravel().astype(np.float32))
            rows.append(dict(entry_id=f"{source['image']}/{row.label}", source_image=source["image"],
                             source_label=int(row.label),
                             plateau=plateau, box_origin=[s.start for s in box], box_shape=list(shape),
                             offset=offset, rim_share=float(row.rim_share), voxels=arrays["voxels"],
                             volume=arrays["voxels"] * voxel, nucleus_voxels=int(row.nucleus_voxels),
                             local_background=[b / plateau for b in arrays["local_background"]],
                             rim_level=[r / plateau for r in arrays["rim_level"]]))
            offset += int(np.prod(shape))
        cut, origins, shares = _patches(source, plateau, cfg, stream)
        patches.extend(cut)
        patch_rows.extend((source["image"], o, s) for o, s in zip(origins, shares))
        image_rows.append(dict(image=source["image"], plateau=plateau, visible_median=vis,
                               plateau_over_median=plateau / median, visible_over_plateau=vis / plateau,
                               **counts, patches=len(cut)))
    entry_count = len(rows)

    def column(key, dtype, width=None):
        values = [r[key] for r in rows]
        if width is None:
            return np.asarray(values, dtype=dtype)
        return np.asarray(values, dtype=dtype).reshape(entry_count, width)

    patch_shape = patches[0].shape if patches else (channels, 0, 0, 0)
    arrays = dict(
        channels=np.asarray(cfg["channels"], dtype=str),
        entry_id=np.asarray([r["entry_id"] for r in rows], dtype=str).reshape(entry_count),
        source_image=np.asarray([r["source_image"] for r in rows], dtype=str).reshape(entry_count),
        source_label=column("source_label", np.int64), plateau=column("plateau", np.float64),
        box_origin=column("box_origin", np.int64, 3), box_shape=column("box_shape", np.int64, 3),
        spacing=np.tile(np.asarray(cfg["spacing_zyx"], dtype=np.float64), (entry_count, 1)),
        offset=column("offset", np.int64), rim_share=column("rim_share", np.float64),
        voxels=column("voxels", np.int64), volume=column("volume", np.float64),
        nucleus_voxels=column("nucleus_voxels", np.int64),
        local_background=column("local_background", np.float64, channels),
        rim_level=column("rim_level", np.float64, channels),
        distance=(np.concatenate(flats) if flats else np.zeros(0)).astype(np.float32, copy=False),
        excess=np.stack([np.concatenate(f) if f else np.zeros(0, dtype=np.float32)
                         for f in excess_flats]).astype(np.float32, copy=False),
        nucleus_distance=(np.concatenate(nucleus_flats) if nucleus_flats else np.zeros(0)).astype(np.float32,
                                                                                                  copy=False),
        images=np.asarray(images, dtype=str), image_plateau=np.asarray(plateaus, dtype=np.float64),
        image_visible_median=np.asarray(visible, dtype=np.float64),
        patch=(np.stack(patches) if patches else np.zeros(patch_shape)).astype(np.float32),
        patch_image=np.asarray([r[0] for r in patch_rows], dtype=str).reshape(len(patch_rows)),
        patch_origin=np.asarray([r[1] for r in patch_rows], dtype=np.int64).reshape(len(patch_rows), 3),
        patch_foreground=np.asarray([r[2] for r in patch_rows], dtype=np.float64))
    totals = {key: int(sum(r[key] for r in image_rows)) for key in _COUNTS}
    record = dict(
        bank=name, appearance_bank_version=APPEARANCE_BANK_VERSION, entry=cfg["entry"], channels=cfg["channels"],
        configuration=cfg, inputs=inputs, held_out=held_out, images=image_rows, plateau_median=median,
        counts=dict(totals, entries=entry_count, patches=len(patches)), digest=_digest(arrays),
        versions=dict(numpy=np.__version__, scipy=scipy.__version__, scikit_image=skimage.__version__))
    return arrays, record


def _plain_name(value, what):
    if not isinstance(value, str) or not _NAME.fullmatch(value.replace("/", "-")):
        raise ValueError(f"{what}: a plain identifier is needed; got {value!r}")
    return value


def _read(spec, crop):
    """One ZYX array of a TIFF file spec (a path, or a mapping with ``path`` and optional ``channel``), cropped to
    ``crop`` (y0, x0, height, width) in Y and X; a YX array gains a singleton Z."""
    from starfinder.io import ImageLoadConfig, load_volume
    if isinstance(spec, (str, os.PathLike)):
        spec = dict(path=spec)
    unknown = sorted(set(spec) - {"path", "channel"})
    if unknown:
        raise ValueError(f"file {spec.get('path')!r}: unknown keys {unknown}")
    path = Path(spec["path"])
    if crop is None:
        image = load_volume(path, config=ImageLoadConfig(channel_index=spec.get("channel"))).image
    else:
        import tifffile
        import zarr
        y0, x0, height, width = (int(v) for v in crop)
        with tifffile.imread(path, aszarr=True) as store:
            array = zarr.open(store, mode="r")
            if spec.get("channel") is not None:
                raise ValueError(f"file {path}: a channel and a crop together are not supported")
            image = np.asarray(array[..., y0:y0 + height, x0:x0 + width])
        if image.ndim == 2:
            image = image[None]
    if image.ndim != 3:
        raise ValueError(f"file {path}: expected a ZYX or YX image; got shape {image.shape}")
    return image, path


def _regions_meet(a, b):
    """Whether two crops (y0, x0, height, width) of one file share a pixel; a whole file (None) meets every crop."""
    if a is None or b is None:
        return True
    (ya, xa, ha, wa), (yb, xb, hb, wb) = ([int(v) for v in a], [int(v) for v in b])
    return ya < yb + hb and yb < ya + ha and xa < xb + wb and xb < xa + wa


def _file_record(path, role, image):
    return dict(image=image, role=role, path=str(Path(path).resolve()), size=Path(path).stat().st_size,
                sha256=file_sha256(path))


def _label_array(labels, what):
    """Labels as signed integers: int32 where every value fits, else int64 (the values are kept)."""
    if not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
        raise ValueError(f"{what}: labels must be non-negative integers")
    narrow = not labels.size or int(labels.max()) <= np.iinfo(np.int32).max
    return labels.astype(np.int32 if narrow else np.int64)


def _source(image, stains, labels, nucleus_labels, spacing, project_z=False):
    """A loaded source image: float64 stains, int64 labels (entry and nucleus) on one grid; the label values kept."""
    stains = [np.asarray(s, dtype=np.float64) for s in stains]
    labels = _label_array(np.asarray(labels), f"source {image}")
    nucleus = None if nucleus_labels is None else _label_array(np.asarray(nucleus_labels), f"source {image}")
    if project_z:
        stains = [s.max(axis=0, keepdims=True) for s in stains]
        labels = labels.max(axis=0, keepdims=True)
        nucleus = None if nucleus is None else nucleus.max(axis=0, keepdims=True)
    shapes = {s.shape for s in stains} | {labels.shape} | ({nucleus.shape} if nucleus is not None else set())
    if len(shapes) != 1:
        raise ValueError(f"source {image}: the stains and labels differ in shape {sorted(shapes)}")
    return dict(image=image, stains=stains, labels=labels, nucleus_labels=nucleus,
                spacing=np.asarray(spacing, dtype=np.float64))


def build_appearance_bank(config: Mapping | str | Path, path: str | Path) -> dict:
    """Build an appearance bank from configured source images and write it; return its build record.

    ``config`` is a mapping or the path of a JSON file (the build configuration, kept with the
    bank outside the repository): ``bank`` (its name), ``channels`` (stain names, the nuclear
    stain first), ``spacing_zyx`` (µm), ``entry`` (``"nucleus"``, or ``"cell"`` for culture
    cells with ``nucleus_labels``), ``sources`` (each with ``image``, ``stains`` mapping each
    channel to a TIFF file, ``labels``, for cells ``nucleus_labels``, an optional ``crop_yx``
    (y0, x0, height, width) and ``project_z``), ``held_out`` (each ``image`` with its
    ``files``, hashed and never a source, and an optional ``crop_yx``: a source may read a
    held-out file only by a crop that shares no pixel with the held-out crop), and optional
    ``keep_z``, ``margin_um``,
    ``selection``, ``fall_off``, ``patches``, ``guard`` and ``seed``. A file is a path or a
    mapping with ``path`` and ``channel``. The rules are those of
    docs/synthetic-real-appearance.md: SEL1 on every label, BR1 with BR4's guard (a refused
    source raises ValueError naming it and its two numbers), OUT1 outside each mask and the
    patch rule of BG1. ``path`` must be ``<bank>.npz``; the bank is written there and its
    record (the configuration, every input with its path, size and SHA-256, the counts per
    rule and image, the plateaus, the digest of the arrays and the file's SHA-256) beside it
    as ``<bank>.record.json``. An existing file raises FileExistsError.
    """
    if not isinstance(config, Mapping):
        config = json.loads(Path(config).read_text(encoding="utf-8"))
    name = config.get("bank")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"bank: a plain file name is needed; got {name!r}")
    path = Path(path)
    if path.name != f"{name}.npz":
        raise ValueError(f"path: the bank {name} is written as {name}.npz; got {path.name}")
    record_path = path.with_name(f"{name}.record.json")
    for target in (path, record_path):
        if target.exists():
            raise FileExistsError(f"bank file exists: {target}")
    cfg = _configuration(config)
    sources_given = config.get("sources")
    if not isinstance(sources_given, (list, tuple)) or not sources_given:
        raise ValueError("sources: at least one source image is needed")
    inputs, sources, seen = [], [], {}
    held_out = []
    for held in config.get("held_out") or []:
        unknown = sorted(set(held) - {"image", "files", "crop_yx"})
        if unknown:
            raise ValueError(f"held_out {held.get('image')!r}: unknown keys {unknown}")
        image = _plain_name(held.get("image"), "held_out image")
        files = [_file_record(f, "held_out", image) for f in held.get("files") or []]
        region = None if held.get("crop_yx") is None else [int(v) for v in held["crop_yx"]]
        held_out.append(dict(image=image, inputs=files) if region is None
                        else dict(image=image, inputs=files, crop_yx=region))
        for f in files:
            seen.setdefault(f["sha256"], []).append((image, region))
    for given in sources_given:
        unknown = sorted(set(given) - {"image", "stains", "labels", "nucleus_labels", "crop_yx", "project_z"})
        if unknown:
            raise ValueError(f"source {given.get('image')!r}: unknown keys {unknown}")
        image = _plain_name(given.get("image"), "source image")
        crop = given.get("crop_yx")
        stain_specs = given.get("stains") or {}
        if sorted(stain_specs) != sorted(cfg["channels"]):
            raise ValueError(f"source {image}: stains {sorted(stain_specs)} do not match channels {cfg['channels']}")
        files = [(stain_specs[c], "stain") for c in cfg["channels"]] + [(given.get("labels"), "labels")]
        if cfg["entry"] == "cell":
            if given.get("nucleus_labels") is None:
                raise ValueError(f"source {image}: cell entries need nucleus_labels")
            files.append((given["nucleus_labels"], "nucleus_labels"))
        arrays = []
        for spec, role in files:
            if spec is None:
                raise ValueError(f"source {image}: no {role} file")
            array, file = _read(spec, crop)
            entry = _file_record(file, role, image)
            for held_image, region in seen.get(entry["sha256"], []):
                if _regions_meet(region, crop):
                    where = "" if region is None else f" (crop_yx {region})"
                    raise ValueError(f"source {image}: {file} is a file of the held-out image {held_image}{where}; "
                                     "a held-out image is never a source")
            inputs.append(dict(entry, channel=spec.get("channel") if isinstance(spec, Mapping) else None,
                               crop_yx=None if crop is None else [int(v) for v in crop]))
            arrays.append(array)
        stains = arrays[:len(cfg["channels"])]
        labels = arrays[len(cfg["channels"])]
        nucleus = arrays[len(cfg["channels"]) + 1] if cfg["entry"] == "cell" else None
        sources.append(_source(image, stains, labels, nucleus, cfg["spacing_zyx"], bool(given.get("project_z"))))
    arrays, record = _build(name, sources, held_out, cfg, inputs)
    record["sources"] = [dict(image=s.get("image"), crop_yx=s.get("crop_yx"), project_z=bool(s.get("project_z")))
                         for s in sources_given]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        np.savez_compressed(handle, **arrays, record=np.asarray(json.dumps(record, sort_keys=True)))
    record = dict(record, generator="starfinder.synthetic.build_appearance_bank",
                  file=dict(path=str(path.resolve()), size=path.stat().st_size, sha256=file_sha256(path)))
    with open(record_path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return record


# ------------------------------------------------------------------------------------------------ closed form


def _closed_form_config(preset, seed):
    """(codebook, config) of a context preset rendered for a closed-form bank: the preset at ``seed`` with noise off
    and without the stains' unevenness field, and without molecules.

    A source image is scaled by one number, its plateau (BR1), and BR4 refuses a source outside a factor 1.5; the
    culture preset's field (amplitude 0.74 over about 50 µm) scales a whole routine scene by more than that (its
    plateaus at the seeds 42 to 44 are 31, 4 and 53 grey levels), so the sources are rendered as one exposure
    (docs/synthetic-real-appearance.md, "The bank without real data", the amendment of 2026-10-17). The
    bank reads no molecule, and the molecule draws are streams of their own, so leaving them out changes neither its
    stains nor its labels."""
    from ._context_presets import context_scene_preset
    from ._observation import NoiseConfig
    book, config = context_scene_preset(preset)
    quiet = NoiseConfig()
    sample = replace(config.sample, molecules_per_cell=0.0, ambient_density=0.0,
                     fov=replace(config.sample.fov, seed=seed, noise=quiet))
    molecules = (None if config.molecules is None
                 else replace(config.molecules, per_cell_median=0.0, ambient_density=0.0))
    stains = tuple(replace(stain, noise=quiet, unevenness=None) for stain in config.stains)
    return book, replace(config, sample=sample, molecules=molecules, stains=stains)


@lru_cache(maxsize=16)
def _closed_form_scene(preset, seed):
    from ._context import generate_context_scene
    book, config = _closed_form_config(preset, seed)
    return generate_context_scene(book, config=config)


def _sample_grid_stain(scene, name):
    """Stain ``name`` of a context scene on its sample grid, by the scene's own stain functions and background, scaled
    to its output and cast as its FOV images are, with no noise."""
    from ._context import _context_model
    from ._observation import _observe, evaluate_background_translated
    from ._stains import _Points
    effective = next(s for s in scene.record["effective_config"]["stains"] if s["name"] == name)
    if effective["noise"]["dependent_enabled"] or effective["noise"]["independent_enabled"]:
        raise ValueError(f"stain {name}: a closed-form source is rendered with its noise off")
    grid = scene.sample.record["sample_grid"]
    shape = tuple(int(v) for v in grid["shape_zyx"])
    plane = _context_model(scene).signal(name, _Points(origin=grid["origin_zyx"], spacing=grid["spacing_zyx"],
                                                       shape=shape))
    background = effective["background"]
    components = background["components"]
    tissue = evaluate_background_translated(components, shape, np.zeros(3))[0] if components else None
    _observe(plane, tissue, background, 0, 0, effective["noise"], None, name, None)
    plane *= effective["scale"]
    return _cast(plane, effective["dtype"]).astype(np.float64)


@lru_cache(maxsize=48)
def _closed_form_stain(preset, seed, name):
    """``_sample_grid_stain`` of a closed-form scene, cached and read-only."""
    image = _sample_grid_stain(_closed_form_scene(preset, seed), name)
    image.setflags(write=False)
    return image


def _context_of(preset):
    from ._context_presets import _PRESET_SPECS
    return _PRESET_SPECS[preset]["context"]


def _closed_form_channels(preset):
    from ._context_presets import context_scene_preset
    names = []
    for stain in context_scene_preset(preset)[1].stains:
        if stain.name not in names:
            names.append(stain.name)
    return names


def _closed_form_source(preset, seed):
    """One source image of a closed-form bank: the scene's sample grid with its stains and truth labels (the 2D
    tissue as the Z maximum of both, its projection truth)."""
    scene = _closed_form_scene(preset, seed)
    context = _context_of(preset)
    stains = [_closed_form_stain(preset, seed, name) for name in _closed_form_channels(preset)]
    labels = scene.sample.sample_labels
    entry_labels = labels["cell"] if context == "culture" else labels["nucleus"]
    nucleus = labels["nucleus"] if context == "culture" else None
    spacing = scene.sample.record["sample_grid"]["spacing_zyx"]
    return _source(f"{preset}/seed-{seed}", stains, entry_labels, nucleus, spacing, project_z=context == "tissue2d")


def _closed_form_configuration(preset):
    context = _context_of(preset)
    from ._context_presets import context_scene_preset
    spacing = context_scene_preset(preset)[1].sample.acquisition.spacing_zyx
    return dict(_CLOSED_FORM_BUILD[context], channels=_closed_form_channels(preset), spacing_zyx=list(spacing))


@lru_cache(maxsize=8)
def _closed_form_bank(preset, n):
    from ._context_presets import CONTEXT_PRESETS
    seed = int(CONTEXT_PRESETS[preset]["seed"])
    seeds = list(range(seed, seed + n))
    name = f"closed-form-{preset}-n{n}"
    cfg = _configuration(_closed_form_configuration(preset))
    sources = [_closed_form_source(preset, s) for s in seeds[:-1]]
    rendering = ("the preset at the seed with noise off, without the stains' unevenness field and without molecules; "
                 "its sample grid's stains and truth labels"
                 + (", as their Z maximum" if _context_of(preset) == "tissue2d" else ""))
    inputs = [dict(image=s["image"], role="source", preset=preset, seed=sd, rendering=rendering)
              for s, sd in zip(sources, seeds)]
    held_out = [dict(image=f"{preset}/seed-{seeds[-1]}", inputs=[dict(role="held_out", preset=preset,
                                                                         seed=seeds[-1])])]
    arrays, record = _build(name, sources, held_out, cfg, inputs)
    record.update(generator="starfinder.synthetic.closed_form_bank", preset=preset, n=n, seeds=seeds)
    return AppearanceBank(arrays, record)


def closed_form_bank(preset: str, n: int = 4) -> AppearanceBank:
    """A bank built in memory from scenes of the accepted closed-form generator (CF1); no file is read or written.

    The context preset ``preset`` (``CONTEXT_PRESETS``) is rendered at ``n`` seeds, the
    preset's seed and the ``n − 1`` after it, with noise off and without the stains'
    unevenness field (one exposure per source image, as BR1 scales it), and, because the bank
    does not read them, without molecules. Each scene's sample
    grid, its stains and its truth labels, is one source image (the 2D tissue as their Z
    maximum); the last seed's image is held out and never a source. The build step of
    ``build_appearance_bank`` runs on them with the context's settings: nucleus entries in
    dense 3D and 2D tissue, cell entries with their nucleus and both stains in the culture.
    Its record names the preset and the seeds instead of the SHA-256 of input files. The
    result is cached per (preset, n) and its arrays are read-only.
    """
    BankReference(preset=preset, n=n)
    return _closed_form_bank(preset, n)


def _load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    record = json.loads(str(arrays.pop("record")))
    return arrays, record


def load_appearance_bank(reference: BankReference) -> AppearanceBank:
    """The bank ``reference`` names: a closed-form bank built in memory, or a stored bank file checked by its SHA-256.

    A stored bank is ``<name>.npz`` in ``reference.directory`` or, when that is None, in the
    directory named by ``STARFINDER_APPEARANCE_BANKS``. ValueError when the variable is unset,
    when the file's SHA-256 differs from ``reference.sha256`` (naming the file and both values)
    or when the arrays do not give the recorded digest; FileNotFoundError when the file is
    missing.
    """
    if not isinstance(reference, BankReference):
        raise TypeError("reference must be BankReference")
    if reference.preset is not None:
        return closed_form_bank(reference.preset, reference.n)
    directory = reference.directory
    if directory is None:
        directory = os.environ.get(BANK_DIRECTORY_VARIABLE)
        if not directory:
            raise ValueError(f"bank {reference.name}: the environment variable {BANK_DIRECTORY_VARIABLE} is not set, "
                             "and the reference names no directory")
    path = Path(directory) / f"{reference.name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"bank {reference.name}: no file {path}")
    sha256 = file_sha256(path)
    if sha256 != reference.sha256:
        raise ValueError(f"bank {reference.name}: the SHA-256 of {path} is {sha256}, not the reference's "
                         f"{reference.sha256}")
    arrays, record = _load_arrays(path)
    digest = _digest(arrays)
    if digest != record["digest"]:
        raise ValueError(f"bank {reference.name}: the arrays of {path} give the digest {digest}, not the recorded "
                         f"{record['digest']}")
    return AppearanceBank(arrays, record, sha256=sha256)
