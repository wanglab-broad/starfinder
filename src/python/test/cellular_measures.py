"""W-393's measures of a nuclear stain and its labels, rebuilt for GN23 (design/validation/synthetic-cellular-scene.md).

Written from the definitions of W-393 ``notes.md`` §2 ("Real measurements", "Bright regions"), not from its scripts;
the functions take arrays and the voxel spacing (µm), so they apply to a real image or to a generated one alike:

* Objects touching the image border (any axis) are left out of the shape quantities (size, ellipsoid fit, solidity)
  and kept for touching and fractions.
* Semi-axes from the second moments of the voxel centres in µm, ``a = sqrt(5 λ)`` in 3D and ``2 sqrt(λ)`` in 2D; the
  ellipsoid fit is the IoU with that moment ellipsoid at the voxel centres of the object's box grown by 25 % on each
  side, "not described by one ellipsoid" below IoU 0.8; solidity is the object's voxels over the voxel centres of its
  box inside the convex hull of its boundary voxel centres.
* Touching: the object shares a face (6-connectivity, 4 in 2D) with another object. Volume fraction: object voxels
  over image voxels.
* Within a nucleus: the CV (sd / mean) of its interior voxels (all 6 neighbours, 4 in 2D, in the same object).
  Between nuclei: each nucleus's median over the median of the nucleus medians of its image (the plateau).
* Edge width 10–90 %: per nucleus the mean stain profile over the signed distance to its boundary (inside the
  distance transform of the nucleus, negative; outside the distance of the background voxels whose nearest object is
  the nucleus, positive) in bins of 0.1 µm from −2 to 2 µm; inside level the mean over −1.5 to −0.8 µm, outside level
  over 0.8 to 1.5 µm; the width from the first 90 % crossing (from −1.5 µm outwards) to the first 10 % crossing after
  it, by linear interpolation; nuclei without both levels or with inside ≤ outside are left out.
* Background: the stain farther than 2 µm from every nucleus, its median over the plateau, per image and per block
  of 128 × 128 pixels (all planes; blocks with at least 2,000 background voxels); noise in the same blocks as
  1.4826 × MAD of ``(I(x + 1) − I(x)) / sqrt(2)`` over pairs of background voxels along X, over the plateau.
* Bright regions: the stain smoothed by a Gaussian of one voxel; a region is a connected component (full
  connectivity) of a nucleus's interior voxels whose smoothed value is at least ``factor`` (1.5) times the median of
  the raw stain over that nucleus, of at least 0.1 µm³ (µm² in 2D); per region its size and its peak raw value over
  the nucleus median; per nucleus the number of regions.
* Reads per cell; the nuclear fraction of reads per cell with at least 10 reads.
* Composite signal share per cell: the share of a cell's pixels above the Otsu threshold of the nuclei–amplicon
  composite (``composite_nuclei_amplicon``).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, Delaunay, QhullError

BINS = np.round(np.arange(-2.0, 2.0 + 1e-9, 0.1), 10)   # bin edges of the edge profile, µm
CENTRES = (BINS[:-1] + BINS[1:]) / 2


def face_structure(ndim):
    """The face neighbourhood: 6 neighbours in 3D, 4 in 2D."""
    return ndimage.generate_binary_structure(ndim, 1)


def touches_border(slices, shape):
    return any(s.start == 0 or s.stop == n for s, n in zip(slices, shape))


def touching_share(labels):
    """(share of the objects that share a face with another object, number of objects)."""
    present = np.unique(labels[labels > 0])
    touching = set()
    for axis in range(labels.ndim):
        a = np.moveaxis(labels, axis, 0)
        x, y = a[:-1], a[1:]
        m = (x > 0) & (y > 0) & (x != y)
        touching |= set(np.unique(x[m]).tolist()) | set(np.unique(y[m]).tolist())
    return (len(touching) / len(present) if len(present) else float("nan")), len(present)


def volume_fraction(labels):
    return float(np.count_nonzero(labels) / labels.size)


def interior(mask):
    """Voxels of ``mask`` whose face neighbours all lie in ``mask`` (a neighbour outside the array does not)."""
    return ndimage.binary_erosion(mask, structure=face_structure(mask.ndim), border_value=0)


def _solidity(mask):
    """Object voxels over the voxel centres of its box inside the convex hull of its boundary voxel centres."""
    boundary = mask & ~interior(mask)
    points = np.argwhere(boundary).astype(np.float64)
    try:
        hull = Delaunay(points[ConvexHull(points).vertices])
    except (QhullError, ValueError):
        return 1.0   # a flat or tiny object: its hull has no volume, every voxel centre is on it
    grid = np.argwhere(np.ones(mask.shape, dtype=bool)).astype(np.float64)
    inside = hull.find_simplex(grid, tol=1e-9) >= 0
    return float(mask.sum() / max(int(inside.sum()), 1))


def object_shapes(labels, spacing):
    """Per object not touching the image border: ID, size (µm³, µm² in 2D), elongation, ellipsoid IoU, solidity."""
    spacing = np.asarray(spacing, dtype=np.float64)
    ndim = labels.ndim
    factor = 5.0 if ndim == 3 else 4.0
    rows = []
    for index, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None or touches_border(slices, labels.shape):
            continue
        # The box grown by 25 % of its extent on each side, clipped to the image.
        grown = tuple(slice(max(0, s.start - int(np.ceil(0.25 * (s.stop - s.start)))),
                            min(n, s.stop + int(np.ceil(0.25 * (s.stop - s.start)))))
                      for s, n in zip(slices, labels.shape))
        mask = labels[grown] == index
        coords = np.argwhere(mask) * spacing
        centre = coords.mean(axis=0)
        if len(coords) < ndim + 1:
            continue
        values, vectors = np.linalg.eigh(np.cov((coords - centre).T))
        semi = np.sqrt(factor * np.clip(values, 0, None))
        if not (semi > 0).all():
            continue
        grid = np.stack(np.meshgrid(*[np.arange(s.stop - s.start) for s in grown], indexing="ij"), -1)
        q = ((grid * spacing - centre) @ vectors) / semi
        ellipsoid = (q * q).sum(axis=-1) <= 1
        union = np.count_nonzero(mask | ellipsoid)
        own = labels[slices] == index
        rows.append(dict(object_id=index, size=float(mask.sum() * np.prod(spacing)),
                         elongation=float(semi[-1] / semi[-2]),
                         iou=float(np.count_nonzero(mask & ellipsoid) / union), solidity=_solidity(own)))
    return rows


def plateau(labels, stain):
    """The image's plateau: the median of its nucleus medians; and the median of each nucleus, by ID."""
    ids = np.unique(labels[labels > 0])
    medians = dict(zip(ids.tolist(), ndimage.median(stain, labels, ids))) if len(ids) else {}
    return (float(np.median(list(medians.values()))) if medians else float("nan")), medians


def _profile(distance, values):
    """The mean stain per 0.1-µm bin of the signed distance (NaN where a bin is empty)."""
    index = np.digitize(distance, BINS) - 1
    keep = (index >= 0) & (index < len(CENTRES))
    sums = np.bincount(index[keep], weights=values[keep], minlength=len(CENTRES))
    counts = np.bincount(index[keep], minlength=len(CENTRES))
    with np.errstate(invalid="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def edge_width(profile):
    """The 10–90 % width of one profile on CENTRES, or None (no level, or inside ≤ outside)."""
    inside = profile[(CENTRES >= -1.5) & (CENTRES <= -0.8)]
    outside = profile[(CENTRES >= 0.8) & (CENTRES <= 1.5)]
    if np.isnan(inside).all() or np.isnan(outside).all():
        return None
    high, low = float(np.nanmean(inside)), float(np.nanmean(outside))
    if not high > low:
        return None
    keep = ~np.isnan(profile) & (CENTRES >= -1.5)
    x, p = CENTRES[keep], (profile[keep] - low) / (high - low)

    def crossing(level, start):
        for i in range(start + 1, len(p)):
            if p[i - 1] >= level > p[i]:
                return i, float(x[i - 1] + (p[i - 1] - level) / (p[i - 1] - p[i]) * (x[i] - x[i - 1]))
        return None, None

    i90, x90 = crossing(0.9, 0)
    if i90 is None:
        return None
    _, x10 = crossing(0.1, i90 - 1)
    return None if x10 is None else x10 - x90


def stain_measures(labels, stain, spacing, ids, factor=1.5, min_size=0.1):
    """Per nucleus of ``ids``: median, interior CV, edge width, bright regions; ``labels`` the image's nuclei.

    Returns (nuclei rows, region rows). The edge profile outside a nucleus uses the background voxels whose nearest
    object is that nucleus; the bright regions use the stain smoothed by a Gaussian of one voxel on the whole image.
    """
    spacing = np.asarray(spacing, dtype=np.float64)
    stain = np.asarray(stain, dtype=np.float64)
    voxel = float(np.prod(spacing))
    background = labels == 0
    outside, nearest = ndimage.distance_transform_edt(background, sampling=spacing, return_indices=True)
    owner = labels[tuple(nearest)]
    smooth = ndimage.gaussian_filter(stain, sigma=1.0)
    structure = np.ones((3,) * labels.ndim, dtype=bool)
    objects = ndimage.find_objects(labels)
    nuclei, regions = [], []
    for n in ids:
        slices = objects[n - 1]
        if slices is None:
            continue
        box = tuple(slice(max(0, s.start - 1), min(d, s.stop + 1)) for s, d in zip(slices, labels.shape))
        own = labels[box] == n
        values = stain[box][own]
        median = float(np.median(values))
        core = interior(own)
        inner = stain[box][core]
        cv = float(inner.std() / inner.mean()) if len(inner) >= 2 and inner.mean() > 0 else None
        # The signed distance: inside the distance transform of the nucleus (to its nearest voxel outside it).
        inside = ndimage.distance_transform_edt(own, sampling=spacing)
        reach = tuple(slice(max(0, s.start - int(np.ceil(2 / h)) - 1), min(d, s.stop + int(np.ceil(2 / h)) + 1))
                      for s, d, h in zip(slices, labels.shape, spacing))
        mine = background[reach] & (owner[reach] == n) & (outside[reach] <= 2.0)
        distance = np.concatenate([-inside[own], outside[reach][mine]])
        profile = _profile(distance, np.concatenate([values, stain[reach][mine]]))
        bright = core & (smooth[box] >= factor * median)
        components, count = ndimage.label(bright, structure=structure)
        found = 0
        for c in range(1, count + 1):
            region = components == c
            size = float(region.sum() * voxel)
            if size < min_size - 1e-12:
                continue
            found += 1
            raw = stain[box][region]
            regions.append(dict(nucleus_id=int(n), size=size, peak=float(raw.max() / median),
                                mean=float(raw.mean() / median), peak_value=float(raw.max())))
        nuclei.append(dict(nucleus_id=int(n), median=median, cv=cv, edge_width=edge_width(profile),
                           regions=found, voxels=int(own.sum())))
    return nuclei, regions


def background_measures(labels, stain, spacing, plateau_value, block=128, min_voxels=2000):
    """(background over plateau of the image, per-block rows of background and noise over plateau)."""
    spacing = np.asarray(spacing, dtype=np.float64)
    stain = np.asarray(stain, dtype=np.float64)
    far = ndimage.distance_transform_edt(labels == 0, sampling=spacing) > 2.0
    level = float(np.median(stain[far]) / plateau_value) if far.any() else None
    blocks = []
    ny, nx = labels.shape[-2:]
    for y0 in range(0, ny, block):
        for x0 in range(0, nx, block):
            window = (..., slice(y0, y0 + block), slice(x0, x0 + block))
            mask, values = far[window], stain[window]
            if mask.sum() < min_voxels:
                continue
            pair = mask[..., 1:] & mask[..., :-1]
            d = ((values[..., 1:] - values[..., :-1]) / np.sqrt(2))[pair]
            noise = float(1.4826 * np.median(np.abs(d - np.median(d)))) if len(d) else None
            blocks.append(dict(y0=y0, x0=x0, background=float(np.median(values[mask]) / plateau_value),
                               noise=None if noise is None else noise / plateau_value))
    return level, blocks


def reads_per_cell(cells, molecules):
    """Reads of each in-sample cell, and the nuclear fraction of reads of each cell with at least 10 reads."""
    kept = cells[cells.in_sample]
    owned = molecules[molecules.true_cell_id.notna()]
    cell = owned.true_cell_id.astype(np.int64)
    total = cell.value_counts()
    nuclear = cell[owned.compartment == "nucleus"].value_counts()
    reads = np.array([int(total.get(int(k), 0)) for k in kept.cell_id])
    fractions = np.array([nuclear.get(int(k), 0) / total[int(k)] for k in kept.cell_id
                          if total.get(int(k), 0) >= 10])
    return reads, fractions


def composite_shares(cells, nuclear, amplicon, ids):
    """Per cell of ``ids``: the share of its pixels above the Otsu threshold of the nuclei–amplicon composite."""
    from skimage.filters import threshold_otsu

    from starfinder.segmentation import composite_nuclei_amplicon

    composite, _ = composite_nuclei_amplicon(nuclear[None], amplicon[None])
    composite = composite[0]
    signal = composite > threshold_otsu(composite)
    return {int(k): float(signal[cells == k].mean()) for k in ids if (cells == k).any()}


# ---------------------------------------------------------------------------------------------- a generated scene

# The target ranges of GN23 (design/validation/synthetic-cellular-scene.md, "Target ranges of the generator checks",
# TR1 to TR12): per quantity the TR row, the kind of object it is measured on (V12) and the range by context.
TARGETS = {
    "touching share": ("TR1", "nuclei", dict(dense3d=(0.44, 0.57), tissue2d=(0.76, 0.82), culture=(0.33, 0.72))),
    "nucleus volume (area) fraction": ("TR2", "nuclei",
                                       dict(dense3d=(0.07, 0.16), tissue2d=(0.30, 0.44), culture=(0.17, 0.45))),
    "share not one ellipsoid (IoU < 0.8)": ("TR3", "nuclei",
                                            dict(dense3d=(0.064, 0.074), tissue2d=(0.027, 0.060),
                                                 culture=(0.12, 0.43))),
    "edge width 10-90 % (µm)": ("TR4", "nuclei", dict(dense3d=(0.20, 2.20), tissue2d=(0.31, 1.57),
                                                      culture=(0.33, 1.63))),
    "bright regions per nucleus": ("TR5", "nuclei", dict(dense3d=(1, 5), tissue2d=(0, 8), culture=(0, 58))),
    "bright region size (µm³; µm² in 2D)": ("TR5", "bright regions",
                                            dict(dense3d=(0.13, 5.04), tissue2d=(0.11, 1.16), culture=(0.11, 7.49))),
    "bright region peak over nucleus median": ("TR5", "bright regions",
                                               dict(dense3d=(1.88, 3.10), tissue2d=(1.76, 2.60),
                                                    culture=(1.90, 9.58))),
    "nucleus median over the image's median": ("TR6", "nuclei",
                                               dict(dense3d=(0.70, 1.44), tissue2d=(0.52, 1.37),
                                                    culture=(0.25, 4.92))),
    "within-nucleus CV": ("TR7", "nuclei", dict(dense3d=(0.25, 0.41), tissue2d=(0.20, 0.38), culture=(0.11, 1.35))),
    "background over plateau, per image": ("TR8", "nuclei", dict(dense3d=(0.01, 0.34), tissue2d=(0.022, 0.031),
                                                                 culture=(0.002, 0.047))),
    "noise over plateau, per 128-px block": ("TR8", "nuclei", dict(dense3d=(0.0, 0.063), tissue2d=(0.0, 0.033),
                                                                   culture=(0.0, 0.24))),
    "8-bit plateau (grey levels)": ("TR9", "nuclei", dict(dense3d=(35, 50), tissue2d=(64, 93), culture=(13, 83))),
    "reads per cell": ("TR10", "molecules", dict(dense3d=(1, 19), tissue2d=(85, 310), culture=(86, 1884))),
    "nuclear fraction of reads per cell": ("TR10", "molecules",
                                           dict(dense3d=(0.25, 0.75), tissue2d=(0.43, 0.93), culture=(0.0, 0.56))),
    "cell volume (expanded nuclei, µm³)": ("TR11", "cells", dict(dense3d=(90, 185))),
    "composite signal share per cell": ("TR12", "cells", dict(tissue2d=(0.52, 0.96))),
}
# TR4 and TR5: measured on context_slow_acquired for 2D tissue and culture, on the preset for dense 3D tissue.
EDGE_AND_REGIONS = ("edge width 10-90 % (µm)", "bright regions per nucleus", "bright region size (µm³; µm² in 2D)",
                    "bright region peak over nucleus median")
MINIMUM = 100   # V12: a range is compared only where the scene holds at least 100 objects of the quantity's kind


def _quantiles(values):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(values):
        return dict(n=0, median=None, p5=None, p95=None)
    return dict(n=int(len(values)), median=float(np.median(values)), p5=float(np.percentile(values, 5)),
                p95=float(np.percentile(values, 95)))


def _measured_once(fov_labels):
    """The FOV each object is measured in: the first FOV, in FOV order, that holds it without touching its Y or X
    border, else the FOV that holds most of its voxels."""
    best = {}
    for k, labels in enumerate(fov_labels):
        counts = np.bincount(labels.reshape(-1).astype(np.int64))
        for index, slices in enumerate(ndimage.find_objects(labels), start=1):
            if slices is None:
                continue
            whole = not any(s.start == 0 or s.stop == n for s, n in zip(slices[-2:], labels.shape[-2:]))
            count, current = int(counts[index]), best.get(index)
            if current is None or (whole and not current[0]) or (not whole and not current[0] and count > current[1]):
                best[index] = (whole, count, k)
    chosen = [[] for _ in fov_labels]
    for index, (_, _, k) in sorted(best.items()):
        chosen[k].append(index)
    return chosen


def scene_targets(scene, context, stain, projection, quantities):
    """W-393's measures of ``quantities`` (keys of TARGETS) on a generated cellular scene.

    Geometry (TR1 to TR3, TR11) on the sample grid's truth labels, as audit A of W-394 measures the touching share
    (``projection``: their Z maximum, the projection truth P); the stain quantities (TR4 to TR9) on each FOV's stain
    ``stain`` and its truth labels (``projection``: both Z maxima), each nucleus measured once (``_measured_once``),
    the plateau and the background per FOV image; TR10 from the truth tables; TR12 on the per-FOV composite of the
    stain and the reference round's channel maximum. Returns one row per quantity: value (the median of its per-object
    or per-image values), p5, p95, the count of objects of its kind, and its range.
    """
    sample = scene.sample
    spacing = np.asarray(sample.record["sample_grid"]["spacing_zyx"], dtype=np.float64)
    planar = spacing[1:] if projection else spacing
    take = (lambda a: a.max(axis=0)) if projection else (lambda a: a)
    values, counts, extra = {}, {}, {}
    nuclei = take(sample.sample_labels["nucleus"]).astype(np.int64)
    if "touching share" in quantities or "nucleus volume (area) fraction" in quantities:
        share, present = touching_share(nuclei)
        values["touching share"], counts["touching share"] = [share], present
        values["nucleus volume (area) fraction"] = [volume_fraction(nuclei)]
        counts["nucleus volume (area) fraction"] = present
    if "share not one ellipsoid (IoU < 0.8)" in quantities:
        shapes = object_shapes(nuclei, planar)
        values["share not one ellipsoid (IoU < 0.8)"] = [
            float(np.mean([r["iou"] < 0.8 for r in shapes])) if shapes else None]
        counts["share not one ellipsoid (IoU < 0.8)"] = len(shapes)
        extra["share not one ellipsoid (IoU < 0.8)"] = dict(
            solidity=_quantiles([r["solidity"] for r in shapes]), iou=_quantiles([r["iou"] for r in shapes]))
    if "cell volume (expanded nuclei, µm³)" in quantities:
        cells = object_shapes(take(sample.sample_labels["cell"]).astype(np.int64), planar)
        values["cell volume (expanded nuclei, µm³)"] = [r["size"] for r in cells]
        counts["cell volume (expanded nuclei, µm³)"] = len(cells)
    stain_keys = set(quantities) & {*EDGE_AND_REGIONS, "nucleus median over the image's median", "within-nucleus CV",
                                    "background over plateau, per image", "noise over plateau, per 128-px block",
                                    "8-bit plateau (grey levels)", "composite signal share per cell"}
    if stain_keys:
        fovs = list(sample.positions.fov_id)
        labels = [take(sample.fov_labels[f]["nucleus"]).astype(np.int64) for f in fovs]
        images = [take(scene.stains[f][stain]) for f in fovs]
        chosen = _measured_once(labels)
        rows, regions, ratios, plateaus, levels, blocks, seen = [], [], [], [], [], [], set()
        for k in range(len(fovs)):
            level, medians = plateau(labels[k], images[k])
            if not medians:
                continue    # a FOV without a nucleus has no plateau
            seen |= set(medians)
            plateaus.append(level)
            nucleus_rows, region_rows = stain_measures(labels[k], images[k], planar, chosen[k])
            rows += nucleus_rows
            regions += region_rows
            ratios += [r["median"] / level for r in nucleus_rows]
            image_level, image_blocks = background_measures(labels[k], images[k], planar, level)
            levels.append(image_level)
            blocks += image_blocks
        top = float(np.iinfo(images[0].dtype).max) if np.issubdtype(images[0].dtype, np.integer) else None
        values.update({
            "edge width 10-90 % (µm)": [r["edge_width"] for r in rows],
            "bright regions per nucleus": [r["regions"] for r in rows],
            "bright region size (µm³; µm² in 2D)": [r["size"] for r in regions],
            "bright region peak over nucleus median": [r["peak"] for r in regions],
            "nucleus median over the image's median": ratios,
            "within-nucleus CV": [r["cv"] for r in rows],
            "background over plateau, per image": levels,
            "noise over plateau, per 128-px block": [b["noise"] for b in blocks],
            "8-bit plateau (grey levels)": plateaus})
        counts.update({
            "edge width 10-90 % (µm)": sum(r["edge_width"] is not None for r in rows),
            "bright regions per nucleus": len(rows),
            "bright region size (µm³; µm² in 2D)": len(regions),
            "bright region peak over nucleus median": len(regions),
            "nucleus median over the image's median": len(rows),
            "within-nucleus CV": sum(r["cv"] is not None for r in rows),
            "background over plateau, per image": len(seen), "noise over plateau, per 128-px block": len(seen),
            "8-bit plateau (grey levels)": len(seen)})
        extra["8-bit plateau (grey levels)"] = dict(
            clipped_plateaus=None if top is None else float(np.mean([r["median"] >= top for r in rows])),
            clipped_region_peaks=None if top is None or not regions else float(
                np.mean([r["peak_value"] >= top for r in regions])),
            nuclei=len(rows), regions=len(regions))
        extra["noise over plateau, per 128-px block"] = dict(blocks=len(blocks),
                                                             background=_quantiles([b["background"] for b in blocks]))
        if "composite signal share per cell" in quantities:
            reference = sample.record["effective_config"]["reference_round"]
            cell_labels = [take(sample.fov_labels[f]["cell"]).astype(np.int64) for f in fovs]
            owners = _measured_once(cell_labels)
            shares = []
            for k, f in enumerate(fovs):
                amplicon = take(sample.fovs[f].rounds[reference].max(axis=-1))
                shares += list(composite_shares(cell_labels[k], images[k], amplicon, owners[k]).values())
            values["composite signal share per cell"] = shares
            counts["composite signal share per cell"] = len(shares)
    if {"reads per cell", "nuclear fraction of reads per cell"} & set(quantities):
        reads, fractions = reads_per_cell(sample.cells, sample.molecules)
        values["reads per cell"], counts["reads per cell"] = list(reads), int(reads.sum())
        values["nuclear fraction of reads per cell"] = list(fractions)
        kept = sample.cells[sample.cells.in_sample].cell_id.astype(np.int64)
        counts["nuclear fraction of reads per cell"] = int(sum(r for r in reads if r >= 10))
        extra["reads per cell"] = dict(cells=int(len(kept)))
        extra["nuclear fraction of reads per cell"] = dict(cells=int(len(fractions)))
    result = []
    for quantity in quantities:
        row, kind, ranges = TARGETS[quantity]
        stats = _quantiles(values[quantity])
        result.append(dict(quantity=quantity, row=row, kind=kind, low=ranges[context][0], high=ranges[context][1],
                           value=stats["median"], p5=stats["p5"], p95=stats["p95"], values=stats["n"],
                           objects=int(counts[quantity]), extra=extra.get(quantity)))
    return result


def outcome(row):
    """V12: in range, outside the range (a finding), or reported (fewer than 100 objects of its kind, or no value)."""
    if row["value"] is None:
        return "reported: no value"
    if row["objects"] < MINIMUM:
        return f"reported: {row['objects']} {row['kind']} < {MINIMUM}"
    return "in range" if row["low"] <= row["value"] <= row["high"] else "outside the range"
