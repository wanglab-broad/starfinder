"""GA23's measures of an appearance scene (docs/synthetic-real-appearance-validation.md, "Target definitions").

The rows TR1 to TR12 are W-393's measures as GN23 rebuilds them (``cellular_measures.scene_targets``), on the truth
labels and the stain of each FOV, each object measured once. The rows the appearance page adds are rebuilt here from
the definitions of W-513 ``scripts/measure.py`` (its docstring), not copied from it:

* **visible-stain share (TR2v)**: the share of an image whose stain, smoothed by a Gaussian of one voxel
  (``scipy.ndimage.gaussian_filter``, sigma 1 on every axis of the array), lies above that smoothed image's Otsu
  threshold (``skimage.filters.threshold_otsu``); the label share of the same form is reported beside it;
* **median outside the visible stain (TR8v)**: the median raw stain over the voxels that are not visible stain, over
  the plateau;
* **background at label rims (TR8r)**: per label not at the border, the 20th percentile of the raw stain over the
  voxels of the label's box, grown by 1.5 µm (``ceil(1.5 / s)`` voxels per axis), that hold no label (at least 21 such
  voxels, else no value), over the plateau; the value is the median over the labels. A dense 3D label at a Z end is at
  the border and its box grows in Z; a culture cell's box spans every plane and only Y and X count as border, as W-513
  measured both;
* **plateau**: the median over the labels of each label's median raw stain (``cellular_measures.plateau``).

TR2v, TR8v and TR8r are measured on the stain of the sample grid, rendered by the generator's point evaluator from
the scene's truth and the bank (the stain DN1 renders), cast by the stain's output as the FOV images are (identity 5:
``clip(rint(g · J), 0, 255)``), with the sample grid's truth labels. Dense 3D is measured on the volume, 2D tissue on
the Z maximum with the projected labels ``P``, the culture's TR2v and TR8v on the Z maximum (DAPI with the projected
nuclei, Flamingo with the projected cells) and its TR8r on the volume around the truth cells, with the plateaus on the
volume (DAPI on the nuclei, Flamingo on the cells).

Beside TR3 the entries the measured truth labels come from are measured the same way (``object_shapes`` on the
entry's mask ``D_e ≤ 0``, padded so it does not touch the array border, at the entry's spacing); beside TR7 each
placed object's CV over its entry's CV (the CV of ``b + E`` over the interior voxels of the entry's mask, the stain
relative to the source plateau), each in the form of its context.

Under the amendment of 2026-10-18 (W-520; operator decision pending Jiahao's confirmation) TR2v of dense 3D and TR8r
of dense 3D and 2D tissue are reported beside their range, not asserted (``AMENDED``).

``load_saved_scene`` reads a scene that ``save_appearance_scene`` wrote back into the attributes these measures and
the point evaluator use, so GA23 measures the scenes GA24 generated in their own processes.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.filters import threshold_otsu
import tifffile

from .cellular_measures import (EDGE_AND_REGIONS, TARGETS, _measured_once, interior, object_shapes,
                                outcome, plateau, scene_targets)

MARGIN_UM = 1.5
RIM_PERCENTILE = 20
RIM_MIN_FREE = 21
# The stain of each context and whether its W-393 quantities are measured on the Z maximum (GN23's CONTEXT).
CONTEXT = {"dense3d": ("DAPI", False), "tissue2d": ("PI", True), "culture": ("DAPI", True)}
# The ranges of the rows the appearance page adds (provisional; "Target definitions"): by context, by stain.
TR2V = {"dense3d": {"DAPI": (0.47, 0.68)}, "tissue2d": {"PI": (0.29, 0.44)},
        "culture": {"DAPI": (0.12, 0.14), "Flamingo": (0.41, 0.53)}}
TR8R = {"dense3d": {"DAPI": (0.33, 0.40)}, "tissue2d": {"PI": (0.125, 0.130)},
        "culture": {"DAPI": (0.0, 0.0), "Flamingo": (0.0, 0.0)}}
TR8V = {"dense3d": {"DAPI": (0.0, 0.54)}, "tissue2d": {"PI": (0.022, 0.047)},
        "culture": {"DAPI": (0.036, 0.038), "Flamingo": (0.022, 0.074)}}
# Rows whose truth-label value the page reports with no range ("none (reported)").
NO_RANGE = {"touching share", "nucleus volume (area) fraction", "background over plateau, per image",
            "cell volume (expanded nuclei, µm³)"}
# Asserted where V12 allows, by context ("In GA23"); every other quantity is reported. TR4 and TR5 of 2D tissue and
# culture are measured on app_slow_acquired and reported.
ASSERTED = {
    "dense3d": {"TR2v", "TR4", "TR5", "TR6", "TR7", "TR8 noise", "TR8r", "TR9", "TR10"},
    "tissue2d": {"TR2v", "TR6", "TR7", "TR8 noise", "TR8r", "TR9", "TR10", "TR12"},
    "culture": {"TR6", "TR8 noise", "TR9", "TR10"},
}
# The amendment of 2026-10-18 (W-520; operator decision pending Jiahao's confirmation, "Target definitions"): these
# rows are reported beside their range, not asserted.
AMENDED = {"dense3d": {"TR2v", "TR8r"}, "tissue2d": {"TR8r"}}
AMENDMENT = "amendment of 2026-10-18 (W-520), beside the range"
COLUMNS = ("preset", "row", "quantity", "value", "low", "high", "objects", "outcome")


def context_of(preset):
    return preset.split("-")[1]


# ------------------------------------------------------------------------------------------------ a saved scene


def _volume(path):
    """A saved image as Z, Y, X (a ZYXC OME-TIFF of one channel squeezed; of several, Z, Y, X, C)."""
    image = np.asarray(tifffile.imread(path))
    if image.ndim == 4:
        image = np.moveaxis(image, 1, -1)
    return image


def load_saved_scene(root, reference_round=True):
    """The scene ``save_appearance_scene`` wrote under ``root``, as the attributes ``scene_targets`` and the point
    evaluator read: ``context.sample`` (record, positions, placement, cells, nuclei, molecules, sample and FOV labels,
    and with ``reference_round`` each FOV's reference-round images), ``context.stains`` and ``context.morphology``,
    ``context.cell_context``, ``placements``, ``background_patches`` and ``record``."""
    root = Path(root)
    record = json.loads((root / "sample.json").read_text(encoding="utf-8"))
    positions = pd.read_csv(root / "positions.csv")
    fov_ids = list(positions.fov_id)
    labels = {}
    for target, name in (("nucleus", "sample_nuclei.tif"), ("cell", "sample_cells.tif")):
        labels[target] = _volume(root / "labels" / name)
    fov_labels, stains, morphology, fovs = {}, {}, {}, {}
    reference = record["effective_config"]["reference_round"]
    for fov in fov_ids:
        base = root / "fovs" / fov
        fov_labels[fov] = {"nucleus": _volume(base / "labels" / "nuclei.tif"),
                           "cell": _volume(base / "labels" / "cells.tif")}
        stains[fov] = {path.name[:-len(".ome.tif")]: _volume(path) for path in sorted((base / "stains").glob("*.tif"))}
        if (base / "morphology.ome.tif").is_file():
            morphology[fov] = _volume(base / "morphology.ome.tif")
        rounds = {reference: _volume(base / "images" / f"{reference}.ome.tif")} if reference_round else {}
        fovs[fov] = SimpleNamespace(rounds=rounds)
    sample = SimpleNamespace(record=record, positions=positions, placement=pd.read_csv(root / "placement_truth.csv"),
                             cells=pd.read_csv(root / "cells.csv"), nuclei=pd.read_csv(root / "nuclei.csv"),
                             molecules=pd.read_csv(root / "molecules.csv"), sample_labels=labels,
                             fov_labels=fov_labels, fovs=fovs)
    context = SimpleNamespace(sample=sample, stains=stains, morphology=morphology,
                              cell_context=pd.read_csv(root / "cell_context.csv"),
                              record=json.loads((root / "context.json").read_text(encoding="utf-8")))
    return SimpleNamespace(context=context,
                           placements=pd.read_csv(root / "placements.csv", dtype={"entry_id": str},
                                                  float_precision="round_trip"),
                           background_patches=pd.read_csv(root / "background_patches.csv",
                                                          dtype={"source_image": str, "stain": str},
                                                          float_precision="round_trip"),
                           record=json.loads((root / "appearance.json").read_text(encoding="utf-8")))


def sample_grid_stains(scene, bank, names):
    """Each stain of ``names`` on the sample grid, rendered by the point evaluator from the scene's truth and the bank
    and cast as the scene's FOV images are (O8: ``clip(rint(g · J), 0, 255)`` for an 8-bit stain)."""
    from starfinder.synthetic._appearance import _point_evaluator
    evaluator = _point_evaluator(scene, bank)
    grid = scene.context.sample.record["sample_grid"]
    g = float(scene.record["plateau"])
    dtypes = {s["name"]: s["dtype"] for s in scene.record["effective_config"]["stains"]}
    out = {}
    for name in names:
        J = evaluator.grid(name, grid["origin_zyx"], grid["shape_zyx"])
        out[name] = np.clip(np.rint(g * J), 0, 255) if dtypes[name] == "uint8" else J
    return out


# ------------------------------------------------------------------------------------------------ the measures


def visible_measures(stain, labels):
    """(visible-stain share, label share, median raw stain outside the visible stain) of an image and its labels."""
    stain = np.asarray(stain, dtype=np.float64)
    smooth = ndimage.gaussian_filter(stain, 1.0)
    visible = smooth > float(threshold_otsu(smooth))
    outside = stain[~visible]
    return (float(visible.mean()), float(np.count_nonzero(labels) / labels.size),
            float(np.median(outside)) if outside.size else None)


def rim_backgrounds(labels, stain, spacing, keep_z):
    """Per label not at the border: the 20th percentile of the raw stain over the unlabelled voxels of its box grown by
    1.5 µm (``keep_z``: the box spans every plane and only Y and X count as border); None where fewer than 21 such
    voxels. ``labels`` and ``stain`` are Z, Y, X (one plane in 2D)."""
    spacing = np.asarray(spacing, dtype=np.float64)
    margin = np.ceil(MARGIN_UM / spacing).astype(int)
    planar = labels.shape[0] == 1
    if planar or keep_z:
        margin[0] = 0
    values = {}
    for index, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        border = any(s.start == 0 or s.stop == n for s, n in zip(slices[1:], labels.shape[1:]))
        if not planar and not keep_z:
            border = border or slices[0].start == 0 or slices[0].stop == labels.shape[0]
        if border:
            continue
        box = tuple(slice(max(s.start - m, 0), min(s.stop + m, n)) for s, m, n in zip(slices, margin, labels.shape))
        if keep_z:
            box = (slice(0, labels.shape[0]),) + box[1:]
        free = labels[box] == 0
        values[index] = (float(np.percentile(np.asarray(stain[box], dtype=np.float64)[free], RIM_PERCENTILE))
                         if free.sum() >= RIM_MIN_FREE else None)
    return values


def _stats(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if not len(values):
        return dict(median=None, p5=None, p95=None, n=0)
    return dict(median=float(np.median(values)), p5=float(np.percentile(values, 5)),
                p95=float(np.percentile(values, 95)), n=int(len(values)))


def _entry_mask(bank, entry, form):
    """An entry's mask (its nucleus in a culture cell) in the form of its context: (mask, spacing)."""
    row = bank.entries.iloc[bank.entry(entry)]
    spacing = np.array([row.spacing_z, row.spacing_y, row.spacing_x], dtype=np.float64)
    d = bank.nucleus_distance(entry) if bank.record["entry"] == "cell" else bank.distance(entry)
    mask = np.asarray(d) <= 0
    if form == "zmax" or mask.shape[0] == 1:
        return mask.max(axis=0), spacing[1:]
    return mask, spacing


def entry_shapes(bank, entries, form):
    """``object_shapes``' IoU of each entry's mask (padded by one voxel), by entry ID."""
    out = {}
    for entry in dict.fromkeys(entries):
        mask, spacing = _entry_mask(bank, entry, form)
        rows = object_shapes(np.pad(mask, 1).astype(np.int64), spacing)
        out[entry] = rows[0]["iou"] if rows else None
    return out


def entry_cv(bank, entry, channel, form):
    """The CV of ``b + E`` (relative to the source plateau) over the interior voxels of the entry's mask, in the form
    of its context (a culture nucleus as Z maxima of the stain and of the mask)."""
    row = bank.entries.iloc[bank.entry(entry)]
    stain = float(row[f"local_background_{channel}"]) + np.asarray(bank.excess(entry, channel), dtype=np.float64)
    mask, _ = _entry_mask(bank, entry, form)
    if form == "zmax" or stain.shape[0] == 1:
        stain = stain.max(axis=0)
    values = stain[interior(mask)]
    return float(values.std() / values.mean()) if len(values) >= 2 and values.mean() > 0 else None


def object_cvs(scene, stain, projection):
    """The within-object CV of each truth nucleus as ``stain_measures`` computes it (each measured once,
    ``_measured_once``), by object ID."""
    sample = scene.context.sample
    take = (lambda a: a.max(axis=0)) if projection else (lambda a: a)
    fovs = list(sample.positions.fov_id)
    labels = [take(sample.fov_labels[f]["nucleus"]).astype(np.int64) for f in fovs]
    chosen = _measured_once(labels)
    out = {}
    for k, f in enumerate(fovs):
        image = np.asarray(take(scene.context.stains[f][stain]), dtype=np.float64)
        objects = ndimage.find_objects(labels[k])
        for n in chosen[k]:
            slices = objects[n - 1]
            box = tuple(slice(max(0, s.start - 1), min(d, s.stop + 1)) for s, d in zip(slices, labels[k].shape))
            inner = image[box][interior(labels[k][box] == n)]
            out[n] = float(inner.std() / inner.mean()) if len(inner) >= 2 and inner.mean() > 0 else None
    return out


# ------------------------------------------------------------------------------------------------ the rows


def _row(preset, row, quantity, value, rng, objects, kind, asserted, reason, extra=None):
    low, high = (None, None) if rng is None else rng
    record = dict(preset=preset, row=row, quantity=quantity, value=value, low=low, high=high, objects=int(objects),
                  kind=kind, extra=extra)
    if asserted and rng is not None:
        record["outcome"] = outcome(record)
    else:
        record["outcome"] = f"reported: {reason}"
    return record


def reported_reason(context, row):
    """Why a row of TR2v or TR8r is reported: the amendment of 2026-10-18, or the page."""
    return AMENDMENT if row in AMENDED.get(context, set()) else f"not asserted for {context} (the page)"


def _tr_key(target_row, quantity):
    if target_row == "TR8":
        return "TR8 noise" if quantity.startswith("noise") else "TR8 background"
    return target_row


def appearance_targets(preset, label, scene, bank, acquired=False):
    """GA23's rows of one scene: ``label`` names it in the table (the preset, ``@acquired`` for a variant)."""
    context = context_of(preset)
    stain, projection = CONTEXT[context]
    asserted = ASSERTED[context] - AMENDED.get(context, set()) if not acquired else set()
    rows = []
    placements = scene.placements
    entry_of = dict(zip(placements.object_id.astype(int), placements.entry_id))
    kind = "cells" if context == "culture" else "nuclei"
    if acquired:
        quantities = list(EDGE_AND_REGIONS)
    else:
        quantities = [q for q, (_, _, ranges) in TARGETS.items() if context in ranges
                      and (context == "dense3d" or q not in EDGE_AND_REGIONS)]
    for r in scene_targets(scene.context, context, stain, projection, quantities):
        key = _tr_key(r["row"], r["quantity"])
        rng = None if r["quantity"] in NO_RANGE else (r["low"], r["high"])
        extra = dict(p5=r["p5"], p95=r["p95"], values=r["values"], **(r["extra"] or {}))
        if acquired:
            reason = "TR4 and TR5 of 2D tissue and culture on app_slow_acquired"
        elif r["quantity"] in NO_RANGE:
            reason = "on the truth labels, no range (the page)"
        elif key == "TR3":
            reason = "the real range reported beside the entries' share (VA4)"
        else:
            reason = f"not asserted for {context} (the page)"
        rows.append(_row(label, r["row"], r["quantity"], r["value"], rng, r["objects"], r["kind"], key in asserted,
                         reason, extra))
        if key == "TR3":
            rows.append(_tr3_entries(label, scene, bank, entry_of, projection, rows[-1]))
        if key == "TR7":
            ratios = []
            cvs = object_cvs(scene, stain, projection)
            form = "zmax" if projection else "volume"
            for k, cv in cvs.items():
                base = entry_cv(bank, entry_of[k], bank.channels[0], form)
                if cv is not None and base:
                    ratios.append(cv / base)
            stats = _stats(ratios)
            rows.append(_row(label, "TR7", "within-nucleus CV over the entry's CV", stats["median"], None,
                             stats["n"], r["kind"], False, "beside TR7 (VA4)", dict(p5=stats["p5"],
                                                                                     p95=stats["p95"])))
    if acquired:
        return rows
    # TR2v, TR8v and TR8r on the sample grid's stain.
    names = list(TR2V[context])
    images = sample_grid_stains(scene, bank, names)
    sample = scene.context.sample
    spacing = np.asarray(sample.record["sample_grid"]["spacing_zyx"], dtype=np.float64)
    nuclei, cells = sample.sample_labels["nucleus"], sample.sample_labels["cell"]
    count = len(placements)
    for name in names:
        image = images[name]
        own = cells if (context == "culture" and name == "Flamingo") else nuclei
        if context == "dense3d":
            form_image, form_labels = image, own
        else:
            form_image, form_labels = image.max(axis=0), own.max(axis=0)
        share, label_share, outside = visible_measures(form_image, form_labels)
        level, _ = plateau(form_labels, form_image)
        rows.append(_row(label, "TR2v", f"visible-stain share ({name})", share, TR2V[context][name], count, kind,
                         "TR2v" in asserted, reported_reason(context, "TR2v"),
                         dict(label_share=label_share, form="volume" if context == "dense3d" else "zmax")))
        rows.append(_row(label, "TR2v", f"label share beside it ({name})", label_share, None, count, kind, False,
                         "the label share beside TR2v (VA2)"))
        rows.append(_row(label, "TR8v", f"median outside the visible stain over plateau ({name})",
                         None if outside is None else outside / level, TR8V[context][name], count, kind, False,
                         "TR8v is reported (VA3)", dict(plateau=level, median_outside=outside)))
        # TR8r: dense 3D on the volume, 2D tissue on the Z maximum, the culture on the volume around the cells.
        if context == "tissue2d":
            rim_labels, rim_image, keep_z = form_labels[None], form_image[None], False
            rim_level = level
        else:
            rim_labels, rim_image, keep_z = (cells if context == "culture" else nuclei), image, context == "culture"
            rim_level, _ = plateau(own, image)
        backgrounds = rim_backgrounds(rim_labels.astype(np.int64), rim_image, spacing, keep_z)
        stats = _stats([None if v is None else v / rim_level for v in backgrounds.values()])
        rows.append(_row(label, "TR8r", f"background at label rims over plateau ({name})", stats["median"],
                         TR8R[context][name], stats["n"], "cells" if context == "culture" else "nuclei",
                         "TR8r" in asserted, reported_reason(context, "TR8r"),
                         dict(plateau=rim_level, p5=stats["p5"], p95=stats["p95"], labels_not_at_border=len(
                             backgrounds))))
    return rows


def _tr3_entries(label, scene, bank, entry_of, projection, placed):
    """The entries' share not one ellipsoid (IoU < 0.8) of the truth labels measured for TR3, and the difference."""
    sample = scene.context.sample
    nuclei = sample.sample_labels["nucleus"]
    labels = (nuclei.max(axis=0) if projection else nuclei).astype(np.int64)
    spacing = np.asarray(sample.record["sample_grid"]["spacing_zyx"], dtype=np.float64)
    ids = [r["object_id"] for r in object_shapes(labels, spacing[1:] if projection else spacing)]
    form = "zmax" if projection else "volume"
    shapes = entry_shapes(bank, [entry_of[k] for k in ids], form)
    values = [shapes[entry_of[k]] for k in ids if shapes[entry_of[k]] is not None]
    share = float(np.mean([v < 0.8 for v in values])) if values else None
    rule4 = bank.record.get("configuration", {}).get("selection", {}).get("rim_share_max")
    difference = None if share is None or placed["value"] is None else placed["value"] - share
    return _row(label, "TR3", "entries' share not one ellipsoid (IoU < 0.8)", share, None, len(values), placed["kind"],
                False, "the entries beside the placed share; the bound on the difference is missing (VA4)",
                dict(placed_share=placed["value"], difference=difference, squash_threshold_rule4=rule4))


def csv_block(rows):
    """The rows as one CSV block (``COLUMNS``)."""
    def cell(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return repr(value)
        text = str(value)
        return f'"{text}"' if any(c in text for c in ',"') else text
    return "\n".join([",".join(COLUMNS)] + [",".join(cell(r[c]) for c in COLUMNS) for r in rows])


def findings(rows):
    """Asserted rows outside their range (V12 allowing)."""
    return [r for r in rows if r["outcome"] == "outside the range"]
