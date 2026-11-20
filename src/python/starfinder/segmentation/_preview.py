"""The preview of input recipes, ``starfinder segment preview`` (design/specs/segmentation-input-recipe.md).

A few FOVs × a few recipes, segmented with one run's method as the workflow would and drawn
on the model input with their size distributions, so that a dataset's recipe can be chosen
before a workflow run. It chooses, ranks and thresholds nothing, runs no preprocessing,
registration, detection or assignment, and writes nothing outside its output folder.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import resource
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ._labels import ReferenceGrid, array_sha256

#: The provisional guard of a preview call: it is not a parameter sweep.
MAX_FOVS = 8
MAX_RECIPES = 8
PREVIEW_FORMAT = "starfinder.segmentation.preview/1"
SIZE_COLUMNS = ("fov", "recipe", "label", "area_um2", "equivalent_diameter_um", "touches_border")
SUMMARY_COLUMNS = ("fov", "recipe", "n_labels", "diameter_p5_um", "diameter_p25_um", "diameter_p50_um",
                   "diameter_p75_um", "diameter_p95_um", "covered_fraction", "n_touching_border")
PERCENTILES = (5, 25, 50, 75, 95)


# --- The configuration --------------------------------------------------------------------------------------

def _configuration(path):
    """The resolved workflow configuration of ``path`` (the plan's shape, schema and resolution) and its SHA-256."""
    import yaml

    from starfinder.workflow._plan import check_shape, resolve, validate_schema
    path = Path(path)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    config = yaml.safe_load(data)
    if not isinstance(config, dict):
        raise ValueError(f"{path} is not a workflow configuration (a YAML mapping)")
    check_shape(config)
    validate_schema(config)
    return resolve(config, {"path": str(path), "sha256": sha})["configuration"], sha


def _recipes(r, names, recipe_file):
    """``[(name, InputRecipe)]`` of the names: recipes of segmentation.input_recipes or of the recipe file."""
    import yaml

    from starfinder._registry import check_name
    from starfinder.dataset.workflow import _input_recipe, _input_recipes
    known = _input_recipes(r)
    if recipe_file is not None:
        values = yaml.safe_load(Path(recipe_file).read_text())
        if not isinstance(values, dict):
            raise ValueError(f"the recipe file {recipe_file} must be a mapping of recipe names to recipes")
        for name, value in values.items():
            check_name(name, "input recipe")
            if name in known:
                raise ValueError(f"recipe {name!r} is in both segmentation.input_recipes and the recipe file "
                                 f"{recipe_file}")
            known[name] = _input_recipe(value, f"{recipe_file}: {name}", name)
    missing = [name for name in names if name not in known]
    if missing:
        raise ValueError(f"the recipes {missing} are not recipes of segmentation.input_recipes"
                         + ("" if recipe_file is None else f" or of {recipe_file}") + f" ({sorted(known)})")
    return [(name, known[name]) for name in names]


def _chain(runs, name, where):
    """The run named ``name`` and the runs its seeds name before it, in order."""
    by_name = {run.name: run for run in runs}
    if name not in by_name:
        raise ValueError(f"run {name!r} is not a run of {where} ({sorted(by_name)})")
    chain = [by_name[name]]
    while chain[0].seeds is not None:
        chain.insert(0, by_name[chain[0].seeds])
    return chain


def _runs(r, run, factor_yx):
    """``(route, runs)``: the previewed run with its seed runs, from segmentation.runs or, with ``factor_yx``, from
    export.segmentation.runs when it has the run."""
    from starfinder.dataset.workflow import _sample_segmentation_block
    from starfinder.workflow import _translate
    export = ((r.get("export") or {}).get("segmentation")) if factor_yx is not None else None
    if export and any(item.get("name") == run for item in export.get("runs") or []):
        config = _sample_segmentation_block(export, dict(r, assignment=None))
        return "after_fusion", _chain(config.runs, run, "export.segmentation.runs")
    if not (r.get("segmentation") or {}).get("runs"):
        raise ValueError(f"run {run!r}: the configuration has no segmentation.runs"
                         + ("" if factor_yx is None else " and no export.segmentation run of that name"))
    plan, _ = _translate.segmentation(r)
    return "per_fov", _chain(plan.runs, run, "segmentation.runs")


# --- The inputs ---------------------------------------------------------------------------------------------

def _source_plan(dataset, fov_id, chain, route):
    """What the FOV's sources need, checked before anything runs: ``(actions, imports)``.

    ``actions`` are ``("reference",)``, ``("registered",)``, ``("saved", round)`` and ``("prepare",)``; a source
    that needs FOV.run or a registration that has not been saved is refused (ValueError naming the workflow step
    that produces it).
    """
    from starfinder import _layout
    from starfinder.dataset.types import REFERENCE_STAIN
    if route == "after_fusion":
        names = list(dict.fromkeys(image for run in chain for _, image in run.inputs))
        return [("image", name) for name in names], _imports(dataset, fov_id, chain)
    fov = dataset.fov(fov_id)
    fov_dir = _layout.fov_dir(dataset.output_root, fov_id)
    actions = []
    sequencing = set(fov.rounds.sequencing_rounds)
    reference = fov._reference_image_path()
    rounds = list(dict.fromkeys(i.round for run in chain for i in run.inputs if i.round is not None))
    merged = any(i.reference_merged for run in chain for i in run.inputs)
    if any(name in sequencing for name in rounds):
        if not (_layout.checkpoint_dir(fov_dir) / _layout.REGISTERED).is_dir():
            raise ValueError(f"FOV {fov_id}: the sequencing rounds {[n for n in rounds if n in sequencing]} are read "
                             "from the fov task's saved registered stage, which is absent: run the fov task with "
                             "processing.checkpoints registered first")
        actions.append(("registered",))
    if Path(reference).is_file():
        actions.append(("reference",))
    else:
        if merged:
            raise ValueError(f"FOV {fov_id}: the reference merged image is read from the saved reference image "
                             f"{reference}, which is absent: run the fov task (the processing block) first")
        if REFERENCE_STAIN not in rounds:
            rounds.insert(0, REFERENCE_STAIN)
    for name in rounds:
        if name in sequencing:
            continue
        saved = _layout.other_round_dir(fov_dir, name) / "image.ome.tif"
        if saved.is_file():
            actions.append(("saved", name))
        elif name == REFERENCE_STAIN:
            actions.insert(0, ("prepare",))
        else:
            raise ValueError(f"FOV {fov_id}: round {name!r} is read from its saved form {saved}, which is absent: "
                             "run the morphology task (the morphology block) first")
    return actions, _imports(dataset, fov_id, chain)


def _imports(dataset, fov_id, chain):
    """The FOV's file of each import run (``{fov_id}`` replaced; relative to the input root); it must exist."""
    from ._import import LabelImportConfig
    files = {}
    for run in chain:
        if isinstance(run.method, LabelImportConfig):
            path = Path(str(run.method.path).replace("{fov_id}", fov_id))
            path = path if path.is_absolute() else Path(dataset.input_root) / path
            if not path.is_file():
                raise ValueError(f"FOV {fov_id}: the label file {path} of import run {run.name!r} does not exist")
            files[run.name] = path
    return files


def _morphology(r):
    """The morphology task's FOV.prepare_morphology settings for the reference stain alone."""
    from starfinder.dataset import MorphologyConfig
    from starfinder.workflow import _translate
    if "morphology" in r:
        return replace(_translate.morphology(r), rounds=())
    return MorphologyConfig(rotation_degrees=_translate.rotation(r))


def _load(dataset, fov_id, actions, r):
    """The FOV with the inputs its actions restore; nothing is written."""
    fov = dataset.fov(fov_id)
    for action in actions:
        if action[0] == "prepare":
            fov.prepare_morphology(_morphology(r))
        elif action[0] == "registered":
            fov.load_checkpoint("registered")
        elif action[0] == "reference":
            fov.load_reference_image()
        elif action[0] == "saved":
            fov.load_registered_round(action[1])
    return fov


def _view_images(dataset, fov, fov_id, names, r):
    """The ZYX image of each export image name of one FOV, read from the files the export reads (the reference
    stain prepared in memory when its saved form is absent), and the FOV's grid."""
    from starfinder.dataset._export import _reference_grid, source_path
    from starfinder.dataset.types import REFERENCE_STAIN
    from starfinder.export._config import image_name
    from starfinder.io import load_volume
    entries = {image_name(entry): entry for entry in (r.get("export") or {}).get("images") or []}
    images = {}
    for name in names:
        if name not in entries:
            raise ValueError(f"image {name!r} is not an image of export.images {sorted(entries)}")
        path = Path(source_path(dataset, fov_id, entries[name]))
        if path.is_file():
            images[name] = np.asarray(load_volume(path).image)
        elif name == REFERENCE_STAIN:
            images[name] = np.asarray(fov.prepare_morphology(_morphology(r)).images[REFERENCE_STAIN])[..., 0]
        else:
            raise ValueError(f"FOV {fov_id}: image {name!r} is read from {path}, which is absent: run the task that "
                             "writes it first")
    return images, _reference_grid(dataset, fov_id)


# --- One FOV and recipe ---------------------------------------------------------------------------------------

def _namespace(dataset, fov_id, run):
    return json.dumps([dataset.dataset_id, dataset.sample_id, fov_id, None, run], separators=(",", ":"))


def _view(images, spacing, k, fov_id):
    """The one-view fused projections of a FOV's source images at factor ``k``, and their grid (``fused_view``)."""
    from starfinder.export._fuse import fuse_one_view
    fused, metadata = {}, None
    for name, image in images.items():
        fused[name], metadata = fuse_one_view(image, spacing, k, name=fov_id)
    shape = next(iter(fused.values())).shape
    return fused, ReferenceGrid(shape, metadata, "declared")


def _segment_view(run, recipe, sources, grid, k, planes, seeds, dataset, fov_id, device):
    """One run on the fused view: its recipe (``k``, the view as the fit region) and its method at factor ``k``."""
    from starfinder.export._segment import scale_lengths

    from ._recipe import apply_input_recipe, check_model_input, model_factor
    from ._segment import segment
    projected = "fused_maximum" if planes > 1 else "single_plane"
    factor = model_factor(None if recipe is None else recipe.model_input, grid.metadata, k=k)
    check_model_input(run.method, recipe, run=run.name, factor_zyx=None if factor is None else factor["factor_zyx"])
    segmentation_input, _, record = apply_input_recipe(sources, grid, recipe=recipe, fit_mask=np.ones(grid.shape_zyx,
                                                                                                      bool),
                                                       k=k, run=run.name, projected=projected)
    used, _ = scale_lengths(run.method, k)
    result = segment(segmentation_input, config=used, target=run.target, seeds=seeds, device=device,
                     label_namespace=_namespace(dataset, fov_id, run.name))
    return result, segmentation_input, record


def _sizes(labels, spacing_yx):
    """Per label of a ZYX label image: its footprint area in µm² (the pixels it covers in Y and X), the equivalent
    diameter ``2·√(area/π)`` and whether it touches the image border; and the fraction of the plane covered."""
    labels = np.asarray(labels)
    _, ny, nx = labels.shape
    z, y, x = np.nonzero(labels)
    values = labels[z, y, x].astype(np.int64)
    if values.size == 0:
        return [], 0.0
    keys = np.unique(values * (ny * nx) + y * nx + x)
    ids, counts = np.unique(keys // (ny * nx), return_counts=True)
    pixel = float(spacing_yx[0]) * float(spacing_yx[1])
    border = np.zeros((ny, nx), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    touching = set(np.unique(labels[:, border]).tolist()) - {0}
    rows = []
    for label, count in zip(ids.tolist(), counts.tolist()):
        area = count * pixel
        rows.append({"label": int(label), "area_um2": area, "equivalent_diameter_um": 2.0 * math.sqrt(area / math.pi),
                     "touches_border": bool(label in touching)})
    covered = float(len(np.unique(keys % (ny * nx)))) / float(ny * nx)
    return rows, covered


def _summary(fov_id, recipe, rows, covered):
    diameters = np.array([row["equivalent_diameter_um"] for row in rows], float)
    values = [float(np.percentile(diameters, p)) if diameters.size else float("nan") for p in PERCENTILES]
    return {"fov": fov_id, "recipe": recipe, "n_labels": len(rows),
            **{f"diameter_p{p}_um": v for p, v in zip(PERCENTILES, values)}, "covered_fraction": covered,
            "n_touching_border": sum(row["touches_border"] for row in rows)}


# --- The figures ------------------------------------------------------------------------------------------------

def _display(image):
    """The plane shown: the first model channel, its Z maximum for a volume."""
    image = np.asarray(image, np.float64)[..., 0]
    return image.max(axis=0)


def _scaled(values):
    low, high = np.percentile(values, (1.0, 99.8))
    return np.clip((values - low) / (high - low), 0, 1) if high > low else np.zeros_like(values)


def _crop(plane, crop_px):
    if crop_px is None:
        return plane
    y0, x0, h, w = crop_px
    return plane[y0:y0 + h, x0:x0 + w]


def _overlay(path, panels, fovs, names, crop_px):
    """overlay.png: per FOV a row of model inputs with label outlines and a row at the model's pixel size; one
    column per recipe."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    from skimage.transform import rescale
    figure, axes = plt.subplots(2 * len(fovs), len(names), figsize=(3.2 * len(names), 3.2 * 2 * len(fovs)),
                                squeeze=False)
    for i, fov in enumerate(fovs):
        for j, name in enumerate(names):
            panel = panels[(fov, name)]
            plane = _crop(_display(panel["image"]), crop_px)
            outline = _crop(find_boundaries(np.asarray(panel["labels"]).max(axis=0), mode="inner"), crop_px)
            rgb = np.repeat(_scaled(plane)[..., None], 3, axis=2)
            rgb[outline] = (1.0, 0.2, 0.2)
            top, bottom = axes[2 * i, j], axes[2 * i + 1, j]
            top.imshow(rgb, interpolation="nearest")
            median = panel["median_um"]
            top.set_title(f"{fov} · {name}\n{panel['n_labels']} labels, median {median:.2f} µm"
                          if math.isfinite(median) else f"{fov} · {name}\n{panel['n_labels']} labels", fontsize=8)
            factor = panel["factor_yx"]
            seen = plane if factor == 1 else rescale(plane, factor, order=1, anti_aliasing=factor < 1)
            bottom.imshow(_scaled(seen), cmap="gray", interpolation="nearest")
            bottom.set_title(f"as the model sees it (factor {factor:g})", fontsize=8)
            for axis in (top, bottom):
                axis.set_xticks([])
                axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)


def _sizes_figure(path, rows, fovs, names):
    """sizes.png: per recipe, the empirical distribution of the equivalent diameter pooled over the FOVs, with one
    thin line per FOV."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.2), squeeze=False)
    for j, name in enumerate(names):
        axis = axes[0, j]
        for fov in fovs:
            values = np.sort([r["equivalent_diameter_um"] for r in rows if r["recipe"] == name and r["fov"] == fov])
            if values.size:
                axis.step(values, np.arange(1, values.size + 1) / values.size, where="post", linewidth=0.6,
                          alpha=0.7, label=fov)
        pooled = np.sort([r["equivalent_diameter_um"] for r in rows if r["recipe"] == name])
        if pooled.size:
            axis.step(pooled, np.arange(1, pooled.size + 1) / pooled.size, where="post", linewidth=2.0, color="black",
                      label="pooled")
        axis.set_title(name, fontsize=9)
        axis.set_xlabel("equivalent diameter (µm)")
        axis.set_ylabel("fraction of labels")
        axis.set_ylim(0, 1)
        axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)


# --- The entry --------------------------------------------------------------------------------------------------

def _crop_pixels(crop_um, spacing_yx):
    if crop_um is None:
        return None
    if len(crop_um) != 4 or any(not math.isfinite(float(v)) for v in crop_um) or crop_um[2] <= 0 or crop_um[3] <= 0:
        raise ValueError(f"crop_um must be four numbers Y, X, H, W in µm with H and W > 0; got {crop_um!r}")
    sy, sx = spacing_yx
    return tuple(max(0, int(math.floor(float(v) / s + 0.5))) for v, s in zip(crop_um, (sy, sx, sy, sx)))


def preview_recipes(config, *, run: str, fovs, recipes, output, recipe_file=None, factor_yx: int | None = None,
                    crop_um=None, device: str = "cpu", save_inputs: bool = False) -> dict:
    """Segment a few FOVs with a few input recipes and draw them, before a workflow run.

    The run's inputs, method, model, seeds and operations are used as
    configured, and only its recipe is replaced by each of ``recipes``; a run
    with seeds also computes its seed run, with that run's own recipe. Each
    source is read from a file the workflow would read (the saved reference
    image, a round's saved form ``other_rounds/<name>/image.ome.tif``, the fov
    task's saved registered stage, an import's file); without its saved form,
    the reference stain is prepared in memory by ``FOV.prepare_morphology``. A
    source that needs ``FOV.run`` or a registration that has not been saved is
    refused before anything runs, naming the workflow step that produces it. No
    preprocessing, registration, detection or assignment runs, and nothing is
    written outside ``output``. Per FOV, the labels and the model input are
    those of ``FOV.segment`` with the recipe (the function it calls).

    With ``factor_yx`` K, the preview is at the pixel size of the route after
    fusion: each FOV's source channels are fused alone at K
    (``starfinder.export._fuse.fuse_one_view``, the call ``fuse`` makes for one
    view, no blend) and the recipe runs with ``k`` K on that ``fused_view``
    grid; the run is taken from ``export.segmentation.runs`` when it has it,
    else from ``segmentation.runs``. A run whose seeds are imported is refused
    with ``factor_yx``: an imported mask lies on another grid.

    Parameters
    ----------
    config : str or pathlib.Path
        The workflow configuration (YAML), with its roots.
    run : str
        The run previewed: a run of ``segmentation.runs`` or, with
        ``factor_yx``, of ``export.segmentation.runs``.
    fovs : sequence of str
        At most 8 FOV ids.
    recipes : sequence of str
        At most 8 names of ``segmentation.input_recipes`` or of
        ``recipe_file``.
    output : str or pathlib.Path
        The output folder; it must be absent or empty.
    recipe_file : str or pathlib.Path or None
        A YAML mapping of recipe names to recipes in the configuration's
        layout, so that a recipe can be tried before it is written into the
        configuration; ``None`` uses the configuration's recipes only.
    factor_yx : int or None
        The fusion factor K of the route after fusion; ``None`` previews on
        each FOV's grid.
    crop_um : tuple of four float or None
        ``(Y, X, H, W)`` in µm of the window the panels show; ``None`` shows
        the whole model input.
    device : str
        ``cpu`` (default) or ``cuda``, passed to the method.
    save_inputs : bool
        Also write each model input as ``inputs/<fov>/<recipe>.ome.tif``.

    Returns
    -------
    dict
        The record written to ``preview.json``: the configuration's path and
        SHA-256, the run, the route and grid, the recipes, and per FOV and
        recipe the recipe record as ``segmentation.json`` holds it, the method
        entry (model artifacts, ``effective``), the model input's and the
        labels' SHA-256, the counts, the wall time and the peak RSS, and the
        files written with their SHA-256.

    Raises
    ------
    ValueError
        Before anything runs: an output folder that is not empty, more than 8
        FOVs or 8 recipes, an unknown run or recipe, a run refused by a recipe,
        a source whose saved form is absent (naming the step that writes it), a
        grid without spacing, or ``factor_yx`` with imported seeds.
    """
    from starfinder._records import software
    from starfinder.dataset.workflow import _workflow_dataset
    from starfinder.workflow._translate import _base

    from ._import import LabelImportConfig
    from ._plan import SegmentationPlan, segment_fov
    from ._recipe import method_value, recipe_config, recipe_sha256

    output = Path(output)
    fovs, names = list(fovs), list(recipes)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"the preview output {output} must be absent or empty")
    if not fovs or not names:
        raise ValueError("the preview needs at least one FOV and one recipe")
    if len(fovs) > MAX_FOVS or len(names) > MAX_RECIPES:
        raise ValueError(f"the preview takes at most {MAX_FOVS} FOVs and {MAX_RECIPES} recipes (a provisional guard: "
                         f"it is not a parameter sweep); got {len(fovs)} FOVs and {len(names)} recipes")
    if len(set(fovs)) != len(fovs) or len(set(names)) != len(names):
        raise ValueError("the FOVs and the recipes of a preview are each given once")
    if factor_yx is not None and (isinstance(factor_yx, bool) or not isinstance(factor_yx, int) or factor_yx < 1):
        raise ValueError(f"factor_yx must be a whole number >= 1; got {factor_yx!r}")
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda'; got {device!r}")
    r, config_sha = _configuration(config)
    selected = _recipes(r, names, recipe_file)
    route, chain = _runs(r, run, factor_yx)
    target, seeds_chain = chain[-1], chain[:-1]
    # Every recipe's run is built (and so checked) before anything runs.
    variants = {name: replace(target, input_recipe=recipe) for name, recipe in selected}
    if factor_yx is not None and any(isinstance(item.method, LabelImportConfig) for item in seeds_chain):
        raise ValueError(f"run {run!r} has imported seeds, which lie on the FOV's grid and not on a fused view: "
                         "preview it without factor_yx")
    dataset = _workflow_dataset(_base(r))
    plans = {fov: _source_plan(dataset, fov, chain, route if factor_yx is not None else "per_fov") for fov in fovs}

    rows, summaries, panels, results = [], [], {}, []
    method_entry = None
    for fov_id in fovs:
        actions, imports = plans[fov_id]
        seed_results = {}
        if factor_yx is None:
            fov = _load(dataset, fov_id, actions, r)
            seed_runs = tuple(replace(item, method=replace(item.method, path=str(imports[item.name])))
                              if item.name in imports else item for item in seeds_chain)
            grid = fov.reference_grid()
            spacing = grid.metadata.spacing_zyx
            if spacing is None:
                raise ValueError(f"FOV {fov_id}: the preview measures sizes in micrometres and needs the grid's "
                                 "spacing: set voxel_size_xy and voxel_size_z")
            if seed_runs:
                fov.segmentation_results.update(segment_fov(fov, SegmentationPlan(seed_runs), device=device))
        else:
            fov = dataset.fov(fov_id)
            if route == "after_fusion":
                images, fov_grid = _view_images(dataset, fov, fov_id, [a[1] for a in actions], r)
            else:
                fov = _load(dataset, fov_id, actions, r)
                from ._plan import _channel
                fov_grid = fov.reference_grid()
                images = {}
                for item in chain:
                    for channel in item.inputs:
                        key = (channel.role, channel.round, channel.channel, channel.reference_merged)
                        images[key] = _channel(fov, channel)[0]
            spacing = fov_grid.metadata.spacing_zyx
            if spacing is None:
                raise ValueError(f"FOV {fov_id}: a fused view needs the grid's spacing: set voxel_size_xy and "
                                 "voxel_size_z")
            fused, view_grid = _view(images, spacing, factor_yx, fov_id)
            planes = fov_grid.shape_zyx[0]
            spacing = view_grid.metadata.spacing_zyx

            def view_sources(item):
                if route == "after_fusion":
                    return [(role, fused[image][0][np.newaxis], {"image": image}) for role, image in item.inputs]
                return [(c.role, fused[(c.role, c.round, c.channel, c.reference_merged)][0][np.newaxis],
                         {"round": c.round, "channel": c.channel, "reference_merged": c.reference_merged})
                        for c in item.inputs]
            for item in seeds_chain:
                seeds = None if item.seeds is None else seed_results[item.seeds]
                seed_results[item.name] = _segment_view(item, item.input_recipe, view_sources(item), view_grid,
                                                        factor_yx, planes, seeds, dataset, fov_id, device)[0]
        for name, recipe in selected:
            started = time.perf_counter()
            if factor_yx is None:
                inputs = {}
                plan = SegmentationPlan((*seed_runs, variants[name]))
                result = segment_fov(fov, plan, device=device, inputs=inputs, only=target.name)[target.name]
                segmentation_input = inputs[target.name]
                record = result.record["input"]["recipe"]
            else:
                seeds = None if target.seeds is None else seed_results[target.seeds]
                result, segmentation_input, record = _segment_view(target, recipe, view_sources(target), view_grid,
                                                                   factor_yx, planes, seeds, dataset, fov_id, device)
                if record["model_input"] is not None:
                    record["model_input"]["method_value"] = method_value(result.record)
            wall = time.perf_counter() - started
            sizes, covered = _sizes(result.labels, spacing[1:])
            for item in sizes:
                rows.append({"fov": fov_id, "recipe": name, **item})
            summary = _summary(fov_id, name, sizes, covered)
            summaries.append(summary)
            factor = segmentation_input.model_scale_zyx
            panels[(fov_id, name)] = {"image": segmentation_input.image, "labels": result.labels,
                                      "n_labels": len(sizes), "median_um": summary["diameter_p50_um"],
                                      "factor_yx": 1.0 if factor is None else factor[1]}
            method = (result.record.get("methods") or [None])[0]
            method_entry = method_entry or method
            results.append({"fov": fov_id, "recipe": name, "recipe_record": record,
                            "method": None if method is None else {k: method.get(k) for k in
                                                                   ("method", "config", "artifacts", "effective")},
                            "model_input": {"sha256": array_sha256(segmentation_input.image),
                                            "shape_zyxc": list(segmentation_input.image.shape),
                                            "dtype": str(segmentation_input.image.dtype),
                                            "model_scale_zyx": None if factor is None else list(factor)},
                            "labels": {"sha256": array_sha256(result.labels), "n_labels": len(sizes),
                                       "shape_zyx": list(result.labels.shape)},
                            "wall_s": wall, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    # Every file below the output folder, written last.
    output.mkdir(parents=True, exist_ok=True)
    crop_px = _crop_pixels(crop_um, spacing[1:])
    _overlay(output / "overlay.png", panels, fovs, names, crop_px)
    _sizes_figure(output / "sizes.png", rows, fovs, names)
    for filename, columns, values in (("sizes.csv", SIZE_COLUMNS, rows), ("summary.csv", SUMMARY_COLUMNS, summaries)):
        with open(output / filename, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(values)
    if save_inputs:
        from starfinder.io import save_volume
        for (fov_id, name), panel in panels.items():
            path = output / "inputs" / fov_id / f"{name}.ome.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_volume(np.asarray(panel["image"]), path, compress=True)
    from starfinder._records import file_sha256
    files = {p.relative_to(output).as_posix(): file_sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    record = {"format": PREVIEW_FORMAT, "config": {"path": str(Path(config).resolve()), "sha256": config_sha},
              "run": run, "route": route, "grid": "fov" if factor_yx is None else "fused_view",
              "factor_yx": factor_yx, "fovs": fovs,
              "recipes": {name: {"sha256": recipe_sha256(recipe), "config": recipe_config(recipe)}
                          for name, recipe in selected},
              "method": method_entry, "results": results, "files": files, "software": software()}
    from starfinder._records import write_json
    write_json(json.loads(json.dumps(record, default=str)), output / "preview.json")
    return record
