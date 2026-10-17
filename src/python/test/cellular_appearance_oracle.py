"""Test oracle of the real-appearance scene (docs/synthetic-real-appearance-validation.md, "Conventions").

The bank's part (W-516): written from the formulas of docs/synthetic-real-appearance.md ("Terms", "The appearance
bank", "What an entry carries", "Composition" for the patch rule) and the bank's record alone. It imports nothing
from ``starfinder.synthetic``; it reads a bank through ``AppearanceBank``'s arrays and record and recomputes, from
the source arrays, which rule leaves out each label, the plateau and the median over the visible stain of each
source image, BR4's guard, and per entry its signed distance, local background, rim level, excess inside the mask and
the four steps of OUT1 outside it, with the library calls the page names (``scipy.ndimage``'s distance transform,
Gaussian filter and median; ``skimage.filters.threshold_otsu``). The settings the page leaves to configuration (the
margins, the fall-off length and blurs, the bins, the patch rule) are read from the bank's record.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

# Labels whose rim share or size lies within this distance of a rule's limit are reported (GA5).
EPS = 1e-9
# One float32 rounding, relative (GA6, GA14).
FLOAT32 = 2.0 ** -23


def face_structure(shape):
    """Face neighbours in 3D; in a single plane only the Y and X neighbours."""
    s = np.zeros((3, 3, 3), dtype=bool)
    s[1, 1, :] = s[1, :, 1] = True
    if shape[0] > 1:
        s[:, 1, 1] = True
    return s


def sigmas(sigma_um, spacing, shape):
    """A Gaussian of ``sigma_um`` µm in source voxels per axis, none along a single-plane axis."""
    return [sigma_um / s if n > 1 else 0.0 for s, n in zip(spacing, shape)]


def signed_distance(mask, spacing):
    """D_e = EDT(not M) − EDT(M) ("Terms")."""
    return ndi.distance_transform_edt(~mask, sampling=spacing) - ndi.distance_transform_edt(mask, sampling=spacing)


def plateau(labels, stain):
    """The median over the labels of each label's median stain ("Terms")."""
    ids = [int(v) for v in np.unique(labels) if v > 0]
    medians = [float(np.median(stain[labels == i])) for i in ids]
    return float(np.median(medians)) if medians else math.nan


def visible_median(stain):
    """The median stain where the stain smoothed by one voxel lies above its Otsu threshold ("Terms")."""
    smooth = ndi.gaussian_filter(stain, [1.0 if n > 1 else 0.0 for n in stain.shape])
    return float(np.median(stain[smooth > threshold_otsu(smooth)]))


def guard(plateaus, visible, factor):
    """BR4: the images refused, each with its reason; and the bank's median plateau."""
    median = float(np.median(plateaus))
    refused = []
    for k, (p, v) in enumerate(zip(plateaus, visible)):
        if p / median > factor or median / p > factor:
            refused.append((k, "median", p, median))
        if v / p > factor or p / v > factor:
            refused.append((k, "visible", p, v))
    return refused, median


def box_of(labels, label, margin_um, spacing, keep_z):
    """The label's bounding box grown by the margin (whole voxels, ceil), clipped; every plane with ``keep_z``."""
    where = np.nonzero(labels == label)
    flat = labels.shape[0] == 1
    margin = [int(math.ceil(m / s)) for m, s in zip(margin_um, spacing)]
    if flat or keep_z:
        margin[0] = 0
    lo = [max(int(w.min()) - m, 0) for w, m in zip(where, margin)]
    hi = [min(int(w.max()) + 1 + m, n) for w, m, n in zip(where, margin, labels.shape)]
    if keep_z:
        lo[0], hi[0] = 0, labels.shape[0]
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def selection(labels, nucleus_labels, spacing, configuration):
    """Per label of a source image: the first rule that leaves it out (``border``, ``size``, ``nucleus``,
    ``squashed``) or ``kept``, its voxels, rim share, nucleus label and nucleus voxels, and the labels within EPS of
    a size or rim-share limit ("Selection of entries", in the rules' order)."""
    rules = configuration["selection"]
    keep_z = configuration["keep_z"]
    flat = labels.shape[0] == 1
    rows = {}
    for label in (int(v) for v in np.unique(labels) if v > 0):
        where = np.nonzero(labels == label)
        z_end = (not flat) and (where[0].min() == 0 or where[0].max() == labels.shape[0] - 1)
        yx_edge = any(w.min() == 0 or w.max() == n - 1 for w, n in zip(where[1:], labels.shape[1:]))
        rows[label] = dict(voxels=len(where[0]), border=yx_edge or (z_end and rules["border_z"]), z_end=z_end)
    sizes = [r["voxels"] for r in rows.values() if not r["border"]]
    lo, hi = np.percentile(sizes, rules["size_percentiles"]) if sizes else (math.nan, math.nan)
    structure = face_structure(labels.shape)
    near = []
    for label, row in rows.items():
        row.update(rim_share=math.nan, nucleus_label=0, nucleus_voxels=0)
        if row["border"]:
            row["outcome"] = "border"
            continue
        if min(abs(row["voxels"] - lo), abs(row["voxels"] - hi)) <= EPS:
            near.append(label)
        if not lo <= row["voxels"] <= hi:
            row["outcome"] = "size"
            continue
        box = box_of(labels, label, configuration["margin_um"], spacing, keep_z)
        own = labels[box] == label
        other = (labels[box] > 0) & ~own
        rim = own & ~ndi.binary_erosion(own, structure=structure)
        row["rim_share"] = float((rim & ndi.binary_dilation(other, structure=structure)).sum() / rim.sum())
        if nucleus_labels is not None:
            inside = nucleus_labels[box][own]
            inside = inside[inside > 0]
            if inside.size:
                ids, counts = np.unique(inside, return_counts=True)
                row["nucleus_label"] = int(ids[np.argmax(counts)])
                row["nucleus_voxels"] = int(counts.max())
            if row["nucleus_voxels"] < rules["min_nucleus_voxels"]:
                row["outcome"] = "nucleus"
                continue
        if abs(row["rim_share"] - rules["rim_share_max"]) <= EPS:
            near.append(label)
        row["outcome"] = "squashed" if row["rim_share"] > rules["rim_share_max"] else "kept"
    counts = dict(labels=len(rows), left_out_border=sum(r["outcome"] == "border" for r in rows.values()),
                  left_out_border_z_alone=(sum(r["z_end"] for r in rows.values())
                                           if rules["border_z"] and not flat else 0),
                  left_out_size_after_border=sum(r["outcome"] == "size" for r in rows.values()),
                  left_out_nucleus_after_size=sum(r["outcome"] == "nucleus" for r in rows.values()),
                  left_out_squashed_after_rest=sum(r["outcome"] == "squashed" for r in rows.values()),
                  kept=sum(r["outcome"] == "kept" for r in rows.values()))
    return rows, counts, near


def fall_off(X, own, other, d, r, spacing, settings, monotone=True):
    """OUT1's four steps on an entry's box ("What an entry carries"): (1) the surroundings smoothed and capped by the
    linear fall-off from the rim level; (2) the weight v of the nearest source label, smoothed, 0 on other labels;
    (3) ``own``, the per-distance median of the capped values where v > 0.5 (made non-increasing when ``monotone``);
    (4) ``kept · v + own · (1 − v)`` blurred outside the mask, the excess unchanged inside; zero beyond λ plus the
    blur's four deviations. Returns the field and the parts (kept, v, own)."""
    length, blur, step = settings["length_um"], settings["blur_um"], settings["bin_um"]
    shape = X.shape
    cap = r * np.maximum(0.0, 1.0 - d / length)
    kept = np.minimum(ndi.gaussian_filter(X, sigmas(settings["smooth_um"], spacing, shape)), cap)
    d_other = ndi.distance_transform_edt(~other, sampling=spacing) if other.any() else np.full(shape, np.inf)
    v = ndi.gaussian_filter((d <= d_other).astype(np.float64), sigmas(settings["weight_um"], spacing, shape))
    v[other] = 0.0
    n = int(round(length / step))
    edges = step * np.arange(n + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    side = ~own & (v > 0.5)
    profile = np.array([np.median(kept[side & (d > a) & (d <= b)])
                        if np.count_nonzero(side & (d > a) & (d <= b)) >= settings["min_bin_voxels"] else np.nan
                        for a, b in zip(edges[:-1], edges[1:])])
    ok = np.isfinite(profile)
    if ok.sum() >= settings["min_bins"]:
        profile = np.interp(centres, centres[ok], profile[ok])
        if monotone:
            profile = np.minimum.accumulate(profile)
        own_field = np.interp(d, np.r_[0.0, centres, length], np.r_[min(r, profile[0]), profile, 0.0])
    else:
        own_field = cap
    outside = kept * v + own_field * (1.0 - v)
    field = np.where(own, X, ndi.gaussian_filter(np.where(own, X, outside), sigmas(blur, spacing, shape)))
    field[d > length + 4 * blur] = 0.0
    return field, dict(kept=kept, v=v, own=own_field)


def entry(stains, labels, nucleus_labels, label, nucleus_label, spacing, configuration, monotone=True):
    """An entry recomputed from its source arrays: box, D_e, nucleus D_e, and per channel the local background ``b``
    (the 20th percentile over the box's unlabelled voxels, 0 with 20 or fewer), the rim level ``r`` (the median
    excess over the inner rim), the excess ``max(I − b, 0)`` and the stored field of OUT1, all in source units."""
    box = box_of(labels, label, configuration["margin_um"], spacing, configuration["keep_z"])
    L = labels[box]
    own = L == label
    other = (L > 0) & ~own
    free = ~own & ~other
    rim = own & ~ndi.binary_erosion(own, structure=face_structure(own.shape))
    d = ndi.distance_transform_edt(~own, sampling=spacing)
    out = dict(box=box, own=own, other=other, d=d, distance=signed_distance(own, spacing), b=[], r=[], X=[], field=[],
               parts=[])
    if nucleus_labels is not None:
        out["nucleus"] = (nucleus_labels[box] == nucleus_label) & own
        out["nucleus_distance"] = signed_distance(out["nucleus"], spacing)
    for stain in stains:
        values = stain[box]
        b = float(np.percentile(values[free], 20)) if np.count_nonzero(free) > 20 else 0.0
        X = np.maximum(values - b, 0.0)
        r = float(np.median(X[rim]))
        field, parts = fall_off(X, own, other, d, r, spacing, configuration["fall_off"], monotone)
        out["b"].append(b), out["r"].append(r), out["X"].append(X), out["field"].append(field)
        out["parts"].append(parts)
    return out


def foreground(labels, nucleus_labels, stains, spacing, plateau_value, settings):
    """BG1's patch rule on a source image: the voxels within ``label_distance_um`` of a label, or with a stain above
    ``level`` of the plateau."""
    labelled = labels > 0
    if nucleus_labels is not None:
        labelled = labelled | (nucleus_labels > 0)
    mask = ndi.distance_transform_edt(~labelled, sampling=spacing) < settings["label_distance_um"]
    for stain in stains:
        mask |= stain > settings["level"] * plateau_value
    return mask


def patch_share(mask, origin, size):
    """The share of a patch's voxels that are foreground."""
    return float(mask[tuple(slice(int(o), int(o) + int(n)) for o, n in zip(origin, size))].mean())


def bank_entries(bank):
    """The (source image, source label) of every entry of a bank."""
    return {(str(i), int(label)) for i, label in zip(bank.arrays["source_image"], bank.arrays["source_label"])}


def stored(bank, entry_id):
    """An entry's stored arrays: (D_e, excess per channel, nucleus D_e or None), float64 views."""
    i = bank.entry(entry_id)
    start, shape = int(bank.arrays["offset"][i]), tuple(int(v) for v in bank.arrays["box_shape"][i])
    size = int(np.prod(shape))
    distance = bank.arrays["distance"][start:start + size].reshape(shape).astype(np.float64)
    excess = bank.arrays["excess"][:, start:start + size].reshape((-1, *shape)).astype(np.float64)
    nucleus = (bank.arrays["nucleus_distance"][start:start + size].reshape(shape).astype(np.float64)
               if bank.record["entry"] == "cell" else None)
    return distance, excess, nucleus
