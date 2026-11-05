"""The API entry of each task kind's command (design/specs/orchestration.md, the table of the commands).

Each function takes the running task (``_task.Run``), reads the resolved configuration through the kind's view,
translates its block with the kept translators and calls the API entry; ``_task.execute`` does the checks, the log,
completion and the records around it.
"""
from __future__ import annotations

import json
import os
from dataclasses import fields, replace
from pathlib import Path

from starfinder import _layout
from starfinder._records import atomic_file
from starfinder.workflow import _translate


def _publish(write, path):
    """Write a file through ``write(tmp)`` to a temporary sibling and rename it into place (OP6)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{os.urandom(8).hex()}.tmp.{path.name}")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _supplied_file(run, value):
    """The supplied statistics file of a recipe: the sample's merged file of the plan's two-pass tasks, else the
    configured file (absolute, or relative to the input root)."""
    from starfinder.workflow._plan import two_pass
    if two_pass(run.configuration.plain()):
        return run.out / str(value).replace("{sample}", run.identity["sample"])
    path = Path(value)
    return path if path.is_absolute() else run.root / path


def _processing(run):
    """(adapted WorkflowConfig with the task's dataset, pipeline with the task's supplied file, stages); a fov task
    reads the supplied statistics file it declares (supplied_input)."""
    translated = _translate.processing(run.configuration.plain())
    adapted = translated.workflow
    run.dataset(adapted.dataset)
    pipeline = adapted.pipeline
    recipe = pipeline.preprocessing
    if recipe is not None and recipe.supplied_statistics is not None:
        path = supplied_input(run) if run.kind in ("fov", "tile_global") else None
        pipeline = replace(pipeline, preprocessing=replace(
            recipe, supplied_statistics=path or _supplied_file(run, recipe.supplied_statistics)))
    return adapted, pipeline, translated.stages


def _declared(run, name):
    """The declared-input form of a supplied file of the manifest's inputs (``input:<path>``, or absolute)."""
    entry = run.manifest["inputs"]["files"].get(name)
    if entry is None:
        return None
    return entry["path"] if Path(entry["path"]).is_absolute() else f"input:{entry['path']}"


def _file(run, name):
    """A supplied file of the manifest's inputs (fovs.csv, the codebook, the position file, ...) by its name."""
    path = Path(run.manifest["inputs"]["files"][name]["path"])
    return path if path.is_absolute() else run.root / path


def codebook_input(run):
    """The codebook (or direct panel) the task reads: its declared input that the manifest names the codebook, or
    None when the task declares none (a fov task that does not decode)."""
    declared = _declared(run, "codebook")
    return run.path(declared) if declared is not None and declared in run.task["inputs"] else None


def supplied_input(run):
    """The supplied statistics file a fov task reads: its declared input that a merge task writes, or the configured
    file the manifest lists; None when it declares none."""
    merged = {o for t in run.manifest["tasks"] if t["kind"] == "merge" for o in t["outputs"]}
    configured = _declared(run, "supplied_statistics")
    found = [i for i in run.task["inputs"] if i in merged or i == configured]
    return run.path(found[-1]) if found else None


def _codebook(run):
    path = codebook_input(run)
    if path is None:
        raise ValueError(f"task {run.id} declares no codebook, which its block reads")
    return path


def fov(run):
    """FOV.run with checkpoints, FOV.save_reference_image and FOV.export_spots (as rsf_single_fov did)."""
    from starfinder.dataset.config import CheckpointConfig
    from starfinder.dataset.workflow import _load_reference
    adapted, pipeline, stages = _processing(run)
    dataset = adapted.dataset
    if pipeline.decoding:
        _load_reference(adapted, _codebook(run))
    dataset._workflow_config = run.out / run.manifest["configuration"]["path"]
    dataset._workflow = run.workflow_section()
    fov_ = dataset.fov(run.identity["fov"])
    execution = adapted.execution
    if run.resume is None:
        fov_.run(pipeline, execution=execution, checkpoints=CheckpointConfig(stages=stages, overwrite=True))
    else:
        _resume(run, fov_, adapted, pipeline, stages)
    if dataset.reference_image_import is None:
        fov_.save_reference_image(projection=adapted.reference_projection, reference_image=adapted.reference_image,
                                  reference_channel=adapted.reference_channel)
    if pipeline.filtering is not None:
        fov_.export_spots()


#: The operations of PipelineConfig after each stage, which a resume from it runs.
_AFTER = {"registered": ("spot_finding", "extraction", "decoding", "scoring", "deduplication", "filtering"),
          "candidates": ("decoding", "scoring", "deduplication", "filtering"), "pre_qc": ("filtering",)}


def _resume(run, fov_, adapted, pipeline, stages):
    """Load the stages up to the resumed one and run the rest of the pipeline (_resume_run); rebuild the reference
    round's image for the reference image when the registered stage, which holds it, is not loaded."""
    from starfinder.dataset.config import CheckpointConfig, PipelineConfig
    stage, record = run.resume
    written = list(record.get(_layout.CHECKPOINTS) or {})
    loaded = [s for s in ("registered", "candidates", "pre_qc") if s in written
              and ("registered", "candidates", "pre_qc").index(s) <= ("registered", "candidates", "pre_qc").index(stage)]
    if stage != "registered":
        loaded = [s for s in loaded if s != "registered"]
    for name in loaded:
        fov_.load_checkpoint(name)
    rest = PipelineConfig(**{f.name: getattr(pipeline, f.name) if f.name in _AFTER[stage] else None
                             for f in fields(PipelineConfig) if f.init})
    later = tuple(s for s in stages if ("registered", "candidates", "pre_qc").index(s)
                  > ("registered", "candidates", "pre_qc").index(stage))
    fov_._resume_run(rest, execution=adapted.execution, checkpoints=CheckpointConfig(stages=later, overwrite=True),
                     stage=stage, record=record)
    if adapted.dataset.reference_image_import is None and stage != "registered":
        # No stage but registered holds the reference round's processed image: it is rebuilt from the raw images
        # of the reference round with the recipe (load, rotation, preprocessing), as FOV.run makes it.
        from starfinder.dataset.types import RoundState
        rounds = adapted.dataset.rounds
        ref = rounds.reference_round
        alone = replace(adapted.dataset, rounds=RoundState(
            sequencing_rounds=[ref] if ref in rounds.sequencing_rounds else [], other_rounds=list(rounds.other_rounds),
            reference_round=ref))
        alone.input_root, alone.output_root = adapted.dataset.input_root, adapted.dataset.output_root
        reference = alone.fov(run.identity["fov"])
        reference.run(PipelineConfig(load=pipeline.load, rotation_degrees=pipeline.rotation_degrees,
                                     preprocessing=pipeline.preprocessing),
                      execution=replace(adapted.execution, mode="batch"), checkpoints=None)
        fov_.images[ref], fov_.metadata[ref] = reference.images[ref], reference.metadata[ref]


def _tiled(run):
    """(adapted WorkflowConfig with the task's dataset and the codebook loaded when the block decodes, pipeline, stages,
    processing.tiled translated, the FOV folder) of a tiled kind."""
    from starfinder.dataset.workflow import _load_reference
    adapted, pipeline, stages = _processing(run)
    if pipeline.decoding:
        _load_reference(adapted, _codebook(run))
    adapted.dataset._workflow_config = run.out / run.manifest["configuration"]["path"]
    adapted.dataset._workflow = run.workflow_section()
    grid = _translate.tiled(run.configuration.plain())
    return adapted, pipeline, stages, grid, _layout.fov_dir(run.out, run.identity["fov"])


def _grid(run, grid, record):
    """The FOV's tiles (tile_grid of the global task's shape), checked against the manifest's tiles of the FOV."""
    from starfinder.dataset.tiled import tile_grid
    tiles = tile_grid(record["shape_zyxc"][1:3], grid.tile_size_yx, grid.halo)
    planned = sorted(t["identity"]["tile"] for t in run.manifest["tasks"] if t["kind"] == "tile"
                     and t["identity"]["fov"] == run.identity["fov"] and t["identity"]["sample"] == run.identity["sample"])
    if sorted(tile.id for tile in tiles) != planned:
        raise ValueError(f"FOV {run.identity['fov']}: the registered rounds' shape {record['shape_zyxc'][:3]} gives the "
                         f"tiles {sorted(tile.id for tile in tiles)}, not the plan's {planned}; plan the run again")
    return tiles


def tile_global(run):
    """run_tile_global (dataset/tiled.py) and FOV.save_reference_image, as direct writes it; a verified resume
    (execution.resume_from_checkpoint, the checkpoint's every round hashed again) has nothing to rerun."""
    from starfinder.dataset.tiled import run_tile_global
    adapted, pipeline, _, grid, folder = _tiled(run)
    if run.resume is not None:
        return
    fov_ = adapted.dataset.fov(run.identity["fov"])
    run_tile_global(fov_, pipeline, execution=adapted.execution, directory=_layout.tiled_dir(folder))
    if adapted.dataset.reference_image_import is None:
        fov_.save_reference_image(projection=adapted.reference_projection, reference_image=adapted.reference_image,
                                  reference_channel=adapted.reference_channel)


def tile(run):
    """run_tile (dataset/tiled.py) of the task's tile: its window of the chunked checkpoint detected with the FOV-level
    thresholds, extracted, decoded and scored; its core's candidates and reads."""
    from starfinder.dataset.tiled import run_tile
    adapted, pipeline, _, grid, folder = _tiled(run)
    tiled = _layout.tiled_dir(folder)
    record = json.loads((tiled / _layout.GLOBAL_RECORD).read_text())
    tiles = {t.id: t for t in _grid(run, grid, record)}
    run_tile(adapted.dataset.fov(run.identity["fov"]), pipeline, tiles[run.identity["tile"]],
             store=tiled / _layout.CHUNKED_CHECKPOINT, statistics=record["statistics"],
             directory=_layout.tile_dir(folder, run.identity["tile"]), halo=grid.halo, exact=grid.exact,
             local_registration=grid.local_registration, post_local_registration=grid.post_local_registration,
             execution=adapted.execution, attempt=run.context.attempt)


def tile_merge(run):
    """merge_tiles (dataset/tiled.py) and FOV.export_spots: the FOV's run.json, molecules.csv and checkpoint stages as
    direct writes them, with run.json's ``tiled`` section; the chunked checkpoint is removed after success unless the
    stage registered is kept."""
    from starfinder._records import file_sha256
    from starfinder.dataset.config import CheckpointConfig
    from starfinder.dataset.tiled import merge_tiles
    from starfinder.workflow._shape import plain
    adapted, pipeline, stages, grid, folder = _tiled(run)
    tiled = _layout.tiled_dir(folder)
    record = json.loads((tiled / _layout.GLOBAL_RECORD).read_text())
    run_record = json.loads((tiled / _layout.RUN_RECORD).read_text())
    tiles = _grid(run, grid, record)
    directories = {t.id: _layout.tile_dir(folder, t.id) for t in tiles}
    kept = "registered" in stages
    entries = []
    for t in tiles:
        tile_record = json.loads((directories[t.id] / "tile.json").read_text())
        entries.append({"tile": t.id, "core_yx": list(t.core_yx), "window_yx": list(t.window_yx),
                        "candidates": tile_record["counts"]["candidates_core"],
                        "reads": tile_record["counts"]["reads_core"],
                        "files": {name: "sha256:" + file_sha256(directories[t.id] / name) for name in _layout.TILE_FILES},
                        "attempt": tile_record["attempt"]})
    detection = record["statistics"]["detection"]
    section = {"grid": {"tile_size_yx": grid.tile_size_yx,
                        "n_yx": [max(t.row for t in tiles) + 1, max(t.col for t in tiles) + 1], "halo": grid.halo,
                        "chunk_zyxc": record["chunk_zyxc"], "local_registration": plain(grid.local_registration)},
               "checkpoint": {"path": f"{_layout.TILED}/{_layout.CHUNKED_CHECKPOINT}",
                              "sha256": {name: entry["sha256"] for name, entry in record["checkpoint"]["rounds"].items()},
                              "kept": kept},
               "tiles": entries,
               "statistics": {"thresholds": None if detection is None else
                              {e["channel"]: e["threshold"] for e in detection["channels"]},
                              "image_background": record["statistics"]["image_background"]}}
    fov_ = adapted.dataset.fov(run.identity["fov"])
    merge_tiles(fov_, pipeline, tiles, tile_directories=directories, global_record=record, global_run_record=run_record,
                execution=adapted.execution, checkpoints=CheckpointConfig(stages=stages, overwrite=True), tiled=section)
    if adapted.dataset.reference_image_import is None:
        fov_._record_output("reference_image", fov_._reference_image_path())
    fov_.export_spots()
    if not kept:
        (tiled / _layout.CHUNKED_CHECKPOINT).unlink(missing_ok=True)


def summarize(run):
    """The summary pass of one supplied step: FOV.run of summary_recipe's recipe, summarize_histograms of each round
    and save_histograms."""
    from starfinder.dataset.config import PipelineConfig
    from starfinder.preprocessing import save_histograms, summarize_histograms, summary_recipe
    adapted, pipeline, _ = _processing(run)
    # A later supplied step is summarized on the earlier step's merged file, the task's declared input (the first
    # step's summary recipe holds no supplied step, and summary_recipe drops the file).
    earlier = run.inputs(_layout.SUPPLIED_FILE)
    recipe = pipeline.preprocessing if not earlier else replace(pipeline.preprocessing, supplied_statistics=earlier[0])
    prefix, after = summary_recipe(recipe, run.identity["step"])
    fov_ = adapted.dataset.fov(run.identity["fov"])
    fov_.run(PipelineConfig(load=pipeline.load, rotation_degrees=pipeline.rotation_degrees, preprocessing=prefix),
             execution=replace(adapted.execution, mode="batch"), checkpoints=None)
    images = {name: fov_.images[name] for name in fov_._pipeline_rounds()}
    summary = summarize_histograms(images, channel_labels=adapted.dataset.channel_order, fov_id=fov_.fov_id,
                                   summarized_after=after)
    (target,) = [path for _, path in run.outputs()]
    _publish(lambda tmp: save_histograms(summary, tmp), target)


def merge(run):
    """The barrier of a supplied step: load_histograms, merge_histograms, fit_supplied_section and
    supplied_statistics with the earlier steps' sections, saved with save_supplied_statistics."""
    from starfinder.preprocessing import (fit_supplied_section, load_histograms, load_supplied_statistics,
        merge_histograms, save_supplied_statistics, step_spec, supplied_statistics)
    adapted, pipeline, _ = _processing(run)
    step = run.identity["step"]
    merged = merge_histograms([load_histograms(path) for path in run.inputs(".npz")])
    (config,) = [s.config for s in pipeline.preprocessing.steps
                 if getattr(s.config, "fit", None) == "supplied" and step_spec(s.config).name == step]
    reference = None if step_spec(config).supplied.per_round else adapted.dataset.rounds.reference_round
    sections = {}
    earlier = [path for path in run.inputs(_layout.SUPPLIED_FILE)]
    if earlier:
        sections.update(load_supplied_statistics(earlier[0])["steps"])
    sections[step] = fit_supplied_section(config, merged, reference_round=reference)
    document = supplied_statistics(merged, sections)
    (target,) = [path for _, path in run.outputs()]
    _publish(lambda tmp: save_supplied_statistics(document, tmp), target)


# --- the morphology, segment and assign kinds (D6) ----------------------------------------------------------------

def _stain_recipe(stain, moving):
    """One translation step on the shared stain: the reference stain channel against the round's channel."""
    from starfinder.registration import (RegistrationRecipe, RegistrationSignalConfig, RegistrationStep,
        TranslationConfig)
    return RegistrationRecipe((RegistrationStep(TranslationConfig()),),
                              signal=RegistrationSignalConfig("channel", stain, moving))


def _prepare_by_stain(fov_, configuration, config, checkpoints):
    """The morphology command's grouping of rounds (OP5), moved unchanged from the nuclei_registration rule.

    Each round is registered by one translation step on its shared stain (its one channel whose name contains
    ref_channel, _nuclei_registration) against the reference stain (_nuclear_stain) as an ExternalReference labelled
    ``<ref_round>:<stain channel>``. The leading rounds that share the first round's stain pattern go through
    FOV.prepare_morphology in one call; a round whose shared stain has another pattern is registered after it by the
    same public calls (load_images with the angle, register), in the configured order.
    """
    import numpy as np
    from starfinder.dataset.config import MorphologyConfig
    from starfinder.dataset.types import REFERENCE_STAIN
    from starfinder.dataset.workflow import _nuclear_stain, _nuclei_registration
    from starfinder.registration import ExternalReference
    from starfinder.workflow._translate import _base
    _, stains, _ = _nuclei_registration(_base(configuration))
    stain = _nuclear_stain(_base(configuration)).channel
    names = list(config.rounds)
    angle, ref = config.rotation_degrees, fov_.rounds.reference_round
    leading = next((i for i, name in enumerate(names) if stains[name] != stains[names[0]]), len(names))
    fov_.prepare_morphology(MorphologyConfig(rotation_degrees=angle, recipe=_stain_recipe(stain, stains[names[0]]),
                                             rounds=tuple(names[:leading])), checkpoints=checkpoints)
    reference = ExternalReference(
        np.ascontiguousarray(fov_.images[REFERENCE_STAIN][..., fov_.dataset.channel_index(REFERENCE_STAIN, stain)]),
        fov_.metadata[REFERENCE_STAIN], f"{ref}:{stain}")
    for name in names[leading:]:
        fov_.load_images(rounds=[name], rotation_degrees=angle)
        fov_.register(_stain_recipe(stain, stains[name]), rounds=[name], reference=reference, checkpoints=checkpoints)


def _channel_images(run, fov_, rounds, projection):
    """``images/<round>/<channel label>/<fov>.tif`` of each channel of the rounds (morphology.channel_images): ZYX,
    or the Z maximum YX with ``projection: max``, keeping the dtype, as the nuclei_registration rule wrote them."""
    from dataclasses import asdict
    import tifffile
    from starfinder.io import save_volume
    from starfinder.preprocessing import ProjectionConfig, project_image
    config = ProjectionConfig() if projection == "max" else None
    for name in rounds:
        for c, channel in enumerate(fov_.dataset.channel_info(name)):
            path = _layout.image_file(run.out, fov_.fov_id, name, channel.channel)
            volume, metadata = fov_.images[name][..., c], fov_.metadata[name]
            if config is None:
                _publish(lambda tmp: save_volume(volume, tmp, metadata=metadata), path)
            else:
                plane = project_image(volume, config=config)[0]
                _publish(lambda tmp: tifffile.imwrite(tmp, plane, photometric="minisblack", metadata={
                    "axes": "YX", "starfinder_metadata": asdict(metadata.projected(method=config.method))}), path)


def morphology(run):
    """FOV.prepare_morphology and FOV.save_registration_record of one FOV (as the nuclei_registration rule did), the
    saved form of every prepared image always written (other_rounds/<name>/), and the channel images with
    morphology.channel_images.

    Without morphology.recipe the rounds are grouped by their shared stain (OP5, _prepare_by_stain); a given recipe
    prepares every round with FOV.prepare_morphology.
    """
    from starfinder.dataset.config import CheckpointConfig
    configuration = run.configuration.plain()
    config = _translate.morphology(configuration)
    fov_ = run.dataset().fov(run.identity["fov"])
    checkpoints = CheckpointConfig(overwrite=True)
    if config.recipe is None:
        _prepare_by_stain(fov_, configuration, config, checkpoints)
    else:
        fov_.prepare_morphology(config, checkpoints=checkpoints)
    _publish(lambda tmp: fov_.save_registration_record(tmp),
             _layout.nuclei_registration_record(run.out, fov_.fov_id))
    images = configuration["morphology"].get("channel_images")
    if images is not None:
        _channel_images(run, fov_, config.rounds, images.get("projection"))


def segment(run):
    """FOV.segment of the task's run with checkpoints into fovs/<fov>/segmentation/<run>/, on its FOV's reference
    grid, after the inputs it declares are restored.

    The grid is the saved reference image (FOV.load_reference_image), or without a processing block the reference
    stain's saved form (OP4); a sequencing round is the fov task's registered stage (FOV.load_checkpoint), every other
    round its saved form (FOV.load_registered_round); the seeds' run is the earlier segment task's
    (FOV.load_segmentation). An import reads the FOV's file that the plan resolved from ``path`` with ``{fov_id}``
    (OP3). The device is segmentation.device, which gives a cuda run's tasks their GPU (``execution.gpu``).
    """
    from starfinder.dataset.config import CheckpointConfig
    from starfinder.dataset.types import REFERENCE_STAIN
    from starfinder.segmentation import LabelImportConfig, SegmentationPlan
    configuration = run.configuration.plain()
    plan_, device = _translate.segmentation(configuration)
    run.device = device
    name = run.identity["run"]
    (current,) = [r for r in plan_.runs if r.name == name]
    if isinstance(current.method, LabelImportConfig):
        # OP3: the run's file of this FOV, the task's declared input.
        (declared,) = [i for i in run.task["inputs"]
                       if i == _declared(run, f"segmentation.{name}.{run.identity['fov']}")]
        current = replace(current, method=replace(current.method, path=run.path(declared)))
        plan_ = SegmentationPlan(tuple(current if r.name == name else r for r in plan_.runs))
    fov_ = run.dataset().fov(run.identity["fov"])
    sequencing = set(fov_.rounds.sequencing_rounds)
    rounds = list(dict.fromkeys(i.round for i in current.inputs if i.round is not None))
    if any(r in sequencing for r in rounds):
        fov_.load_checkpoint("registered")
    reference = next((i for i in run.task["inputs"] if run.path(i) == fov_._reference_image_path()), None)
    if reference is not None:
        fov_.load_reference_image()
    elif REFERENCE_STAIN not in rounds:
        rounds.insert(0, REFERENCE_STAIN)
    for other in rounds:
        if other not in sequencing:
            fov_.load_registered_round(other)
    if current.seeds is not None:
        fov_.load_segmentation(current.seeds)
    fov_.segment(plan_, device=device, checkpoints=CheckpointConfig(overwrite=True), run=name)


def _codebook_genes(run, dataset):
    """The genes of the count matrix in their order, from the codebook the task declares: the direct panel's
    (DirectPanel.genes) in readout_mode direct, else the codebook's gene column in first-appearance order
    (Codebook.genes), which no encoding changes, so that the assign kind reads no processing block."""
    import csv
    path = _codebook(run)
    if dataset.readout_mode == "direct":
        dataset.load_direct_panel(path)
        return tuple(dataset.direct_panel.genes)
    genes, column = [], 0
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for line, values in enumerate(csv.reader(handle), 1):
            if line == 1 and values and values[0].strip() in ("gene", "gene_id", "entry_id"):
                header = [v.strip() for v in values]
                column = header.index("gene_id" if "gene_id" in header else "gene")
                continue
            if values:
                genes.append(values[column].strip())
    return tuple(dict.fromkeys(genes))


def assign(run):
    """assign_molecules of one FOV's molecule table on its saved segmentation runs, saved as
    fovs/<fov>/assignment/<name>/ (as the reads_assignment rule did with an assembly block), and the FOV's previews
    with assignment.previews.

    The molecules are fovs/<fov>/molecules.csv (load_molecules, their identities kept); the grid is the FOV's
    reference image (FOV.load_reference_image); the cell and nucleus runs are the segment tasks' saved runs
    (FOV.load_segmentation), which the saved assignment links; the record holds summarize_assignment's counts.
    """
    from starfinder.assignment import assign_molecules, load_molecules, summarize_assignment
    from starfinder.assignment._persist import assignment_directory, write_assignment
    from starfinder.dataset.workflow import _label_namespace
    from starfinder.segmentation._persist import saved_labels
    from starfinder.workflow._previews import assignment_figures
    from starfinder.workflow._translate import _base
    configuration = run.configuration.plain()
    config, call = _translate.assignment(configuration)
    name = call.get("name", "default")
    if call.get("population", "final") != "final":
        raise ValueError(f"assignment.population {call['population']!r}: the assign task reads the FOV's "
                         "molecules.csv, which holds the final population only")
    dataset = run.dataset()
    fov_id = run.identity["fov"]
    genes = _codebook_genes(run, dataset)
    fov_ = dataset.fov(fov_id)
    fov_.load_reference_image()
    grid = fov_.reference_grid()
    cells = fov_.load_segmentation(call.get("cells", "cell"))
    nuclei = (None if call.get("nuclei") is None else cells if call["nuclei"] == call.get("cells", "cell")
              else fov_.load_segmentation(call["nuclei"]))
    molecules = load_molecules(_layout.spot_table(run.out, fov_id), genes=genes,
                               spot_namespace=_label_namespace(_base(configuration), fov_id))
    result = assign_molecules(molecules, cells, grid=grid, nuclei=nuclei, config=config)
    result = replace(result, record=dict(result.record, name=name))
    result = replace(result, record=dict(result.record, counts=summarize_assignment(result)))
    folder = _layout.fov_dir(run.out, fov_id)
    result = write_assignment(assignment_directory(folder, name), result, metadata=cells.grid.metadata,
                              cells_saved=saved_labels(cells, folder),
                              nuclei_saved=None if nuclei is None else saved_labels(nuclei, folder),
                              same_run=nuclei is cells)
    if run.api["assignment"]["previews"]:
        assignment_figures(result, fov_._reference_image_path(), _layout.fov_previews_dir(run.out, fov_id))


def _stitching(run):
    """(block, dataset, membership, FOVs, PositionIDs, grid FOVs) of a stitching kind, the position file the
    manifest names in place of the configured path."""
    from starfinder.dataset.workflow import _membership, _sample_grid_fovs, _stage_position_ids
    configuration = run.configuration.plain()
    block = _translate.stitching(configuration)
    source = block.positions.get("source")
    if source in ("table", "stage_file") and "positions" in run.manifest["inputs"]["files"]:
        block = replace(block, positions=dict(block.positions, **{source: str(_file(run, "positions"))}))
    dataset = run.dataset()
    fovs_csv = _file(run, "fovs.csv")
    membership = _membership(configuration, fovs_csv)
    sample = run.identity["sample"]
    fov_ids = list(run.manifest["membership"]["samples"][sample])
    position_ids = _stage_position_ids({"stitching": configuration["stitching"]}, membership)
    grid_fov_ids = _sample_grid_fovs(configuration, sample, fovs_csv)
    return block, dataset, fov_ids, position_ids, grid_fov_ids


def _grids(run, dataset, fov_ids):
    from starfinder.dataset._declared import reference_grid, reference_settings
    settings = reference_settings(run.configuration.plain())
    return {fov: reference_grid(dataset.output_root, fov, **settings) for fov in fov_ids}


def stitch(run):
    """Dataset.stitch of one sample into stitching/<sample>/ (an in-process method)."""
    from starfinder.dataset.workflow import _sample_table
    block, dataset, fov_ids, position_ids, grid_fov_ids = _stitching(run)
    table = block.positions
    if table.get("source") == "regular_grid" and grid_fov_ids != fov_ids:
        table = _sample_table(block, dataset, run.identity["sample"], fov_ids, _grids(run, dataset, fov_ids),
                              position_ids, grid_fov_ids)
    dataset.stitch(run.identity["sample"], fov_ids, table=table, config=block.config, input=block.input,
                   review=block.review, fallback=block.fallback,
                   output_dir=_layout.stitching_dir(run.out, run.identity["sample"]), position_ids=position_ids)
    _keep_approvals(run, block.review)


def earlier_approvals(run):
    """The approvals of the sample's earlier ``placement.json`` (``review.approvals``), read by the preparation of a
    placement task before its outputs are removed; () when there is none or it cannot be read."""
    path = _layout.stitching_dir(run.out, run.identity["sample"]) / "placement.json"
    try:
        approvals = (json.loads(path.read_text()).get("review") or {}).get("approvals")
    except (OSError, ValueError, AttributeError):
        return ()
    return tuple(approvals) if isinstance(approvals, list) else ()


def _keep_approvals(run, policy):
    """Keep the earlier approvals (``run.approvals``) whose table SHA-256 equal the new placement's, the rule of
    review_placement, and save the placement with them: an approval saved with save_placement survives a rerun of
    the task that places the FOVs as before. Without a kept approval the saved placement stays as written."""
    if not run.approvals:
        return
    import logging
    from starfinder.stitching import ReviewPolicy, load_placement, review_placement, save_placement
    folder = _layout.stitching_dir(run.out, run.identity["sample"])
    result = load_placement(folder)
    review = dict(result.record.get("review") or {}, approvals=list(run.approvals))
    kept = review_placement(replace(result, record=dict(result.record, review=review)),
                            ReviewPolicy() if policy is None else policy)
    approvals = kept.record["review"]["approvals"]
    logging.getLogger("starfinder.workflow").info(
        f"kept {len(approvals)} of the {len(run.approvals)} approvals of the earlier placement (those whose table "
        "SHA-256 equal the new tables')")
    if approvals:
        save_placement(kept, folder)


def bigstitcher_macro(run):
    """export_bigstitcher of one sample: the tiles, the macro and the view map in <save dir>/<sample>/."""
    from starfinder.dataset._declared import reference_settings
    from starfinder.dataset._stitching import read_inputs
    from starfinder.dataset.workflow import _sample_table
    from starfinder.stitching import export_bigstitcher
    block, dataset, fov_ids, position_ids, grid_fov_ids = _stitching(run)
    folder = block.bigstitcher_save_dir / run.identity["sample"]
    loaded = read_inputs(dataset.output_root, fov_ids, block.input,
                         reference=reference_settings(run.configuration.plain()))
    table = _sample_table(block, dataset, run.identity["sample"], fov_ids,
                          {fov: item.grid for fov, item in loaded.items()}, position_ids, grid_fov_ids)
    text, view_map = export_bigstitcher(table, [loaded[fov] for fov in table.fov_ids], config=block.config,
                                        tile_dir=folder / "tiles", save_dir=folder)
    with atomic_file(folder / "BigStitcher_macro.ijm") as tmp:
        tmp.write_text(text)
    with atomic_file(folder / "view_map.csv") as tmp:
        view_map.to_csv(tmp, index=False)


def _read_view_map(path):
    import pandas as pd
    return pd.read_csv(path, dtype={"fov_id": str}, keep_default_na=False)


def _check_dataset_xml(dataset_xml, view_map):
    """_check_bigstitcher_engine: dataset.xml exists and holds a Stitching Transform for every view of the map."""
    from xml.etree import ElementTree

    from starfinder.stitching._bigstitcher import (STITCHING_TRANSFORM, _children, _setup_tiles, _text)
    path = Path(dataset_xml)
    if not path.is_file():
        raise RuntimeError(f"the BigStitcher engine task wrote no dataset.xml at {path}")
    try:
        root = ElementTree.parse(path).getroot()
        tiles = _setup_tiles(root, path)
    except (ElementTree.ParseError, ValueError) as error:
        raise RuntimeError(f"{path} is not a dataset.xml the import can read: {error}") from None
    placed = set()
    for registration in _children(root, "ViewRegistration"):
        setup = registration.get("setup")
        names = {_text(transform, "Name") for transform in _children(registration, "ViewTransform")}
        if setup is not None and setup.isdigit() and int(setup) in tiles and STITCHING_TRANSFORM in names:
            placed.add(tiles[int(setup)])
    missing = [int(view) for view in view_map["view"] if int(view) not in placed]
    if missing:
        raise RuntimeError(f"{path} holds no {STITCHING_TRANSFORM!r} for views {', '.join(map(str, missing))}; the "
                           "BigStitcher engine task failed (Fiji exits 0 when its macro fails)")


def _pair(run, block):
    """(dataset.xml, view map, macro or None) of a sample: the supplied pair (D3), else the macro and engine
    tasks' files in <save dir>/<sample>/."""
    sample = run.identity["sample"]
    if f"dataset_xml.{sample}" in run.manifest["inputs"]["files"]:
        return _file(run, f"dataset_xml.{sample}"), _file(run, f"view_map.{sample}"), None
    folder = block.bigstitcher_save_dir / sample
    return folder / "dataset.xml", folder / "view_map.csv", folder / "BigStitcher_macro.ijm"


def bigstitcher_check(run):
    """The check after Fiji: dataset.xml against the view map (a failed check removes dataset.xml, exit 1)."""
    (dataset_xml,) = [path for _, path in run.outputs()]
    (view_map,) = run.inputs("view_map.csv")
    _check_dataset_xml(dataset_xml, _read_view_map(view_map))


def bigstitcher_import(run):
    """The import of dataset.xml (checked first) with the view map and the macro, saved with save_placement."""
    from starfinder.dataset.workflow import _sample_table
    from starfinder.stitching import import_bigstitcher, save_placement
    block, dataset, fov_ids, position_ids, grid_fov_ids = _stitching(run)
    dataset_xml, view_map_path, macro = _pair(run, block)
    view_map = _read_view_map(view_map_path)
    _check_dataset_xml(dataset_xml, view_map)
    grids = _grids(run, dataset, fov_ids)
    table = _sample_table(block, dataset, run.identity["sample"], fov_ids, grids, position_ids, grid_fov_ids)
    result = import_bigstitcher(dataset_xml, table=table, view_map=view_map, grids=grids, config=block.config,
                                review=block.review, fallback=block.fallback, sample=run.identity["sample"],
                                macro=None if macro is None else Path(macro).read_text())
    save_placement(result, _layout.stitching_dir(run.out, run.identity["sample"]))
    _keep_approvals(run, block.review)


def _from_plain(value):
    """A plain value of the resolved configuration back as the dataclass its ``type`` names (the encoding and layout
    of ``api.processing.codebook``, which the export loads the codebook with)."""
    from starfinder.readout import BarcodeLayout, EncodingConfig, OneBaseEncodingConfig, Segment
    types = {cls.__name__: cls for cls in (BarcodeLayout, EncodingConfig, OneBaseEncodingConfig, Segment)}
    if isinstance(value, dict) and "type" in value:
        cls = types[value["type"]]
        return cls(**{f.name: _from_plain(value[f.name]) for f in fields(cls) if f.init and f.name in value})
    if isinstance(value, dict):
        return {k: _from_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return tuple(_from_plain(v) for v in value)
    return value


def export(run):
    """Dataset.export of one sample into export/<sample>/; the molecule routes load the codebook for its genes.

    With export.previews on the route per FOV with ``assemble`` (the plan then declares export/<sample>/previews/),
    each assembled FOV's assignment figures with its ownership, into previews/<fov>/ (P5), over the FOV's reference
    image when it is a declared input; the folder exists, empty, when the assemble part was neither written nor
    reused. Before Dataset.export, the command removes its earlier previews/ whatever export.previews says;
    no export record lists that folder. A copy of export.json under workflow/ survives engine removal of the
    declared record, so Dataset.export can remove earlier listed files and reuse saved assembly on a rerun.

    A placement that needs review and holds no approval of its tables holds the export (TaskHeld, exit 4): after
    its declared outputs are removed, the task makes the sample report (_held_report) for the reviewer when every
    other task the report needs has succeeded.
    """
    import shutil
    from starfinder.export._record import PARTS, listed_files
    from starfinder.stitching import PlacementReviewRequiredError
    from starfinder.workflow._task import TaskHeld

    config, assembly = _translate.export(run.configuration.plain())
    dataset = run.dataset()
    export_api = run.api["export"]
    if export_api["route"] != "per_fov":
        # The molecule routes read each FOV's molecule table with the codebook's genes, loaded with the processing
        # block's encoding and layout (a block the export's map names).
        if dataset.readout_mode == "direct":
            dataset.load_direct_panel(_codebook(run))
        else:
            codebook = run.api["processing"]["codebook"]
            dataset.load_codebook(_codebook(run), encoding=_from_plain(codebook["encoding"]),
                                  layout=_from_plain(codebook["layout"]))
    sample = run.identity["sample"]
    previews = _layout.export_previews_dir(run.out, sample)
    if previews.exists():
        shutil.rmtree(previews)
    folder = _layout.export_dir(run.out, sample)
    record_path = folder / _layout.EXPORT_FILE
    saved = _layout.workflow_export_record(run.out, sample)
    if not record_path.exists() and saved.is_file():
        previous = json.loads(saved.read_text())
        # A removed folder (even recreated empty or with foreign files) has no saved assembly to reuse:
        # do not resurrect its record and turn the API's "no saved assembly" refusal into a missing-table error.
        if any((folder / name).exists() for part in PARTS for name in listed_files(previous, part)):
            with atomic_file(record_path) as tmp:
                shutil.copyfile(saved, tmp)
    try:
        record = dataset.export(sample, placement=_layout.stitching_dir(run.out, sample),
                                assignment=assembly.assignment, config=config, output_dir=folder)
    except PlacementReviewRequiredError as error:
        raise TaskHeld(f"{error}; then save_placement it into {_layout.stitching_dir(run.out, sample)} and run the "
                       "workflow again", then=lambda: _held_report(run)) from error
    finally:
        # _task removes declared outputs only after the command raises; keep a failed attempt's record too.
        # SIGKILL runs no finally: files listed only by the killed attempt can outlive this earlier copy.
        if record_path.is_file():
            with atomic_file(saved) as tmp:
                shutil.copyfile(record_path, tmp)
    if f"{Path(os.path.relpath(previews, run.out)).as_posix()}/" not in run.task["outputs"]:
        return
    from starfinder.workflow._previews import ownership_figures
    previews.mkdir(parents=True, exist_ok=True)
    if record.parts.get("assemble") in ("written", "reused"):
        declared = set(run.inputs())
        images = {fov: path for fov in run.manifest["membership"]["samples"][sample]
                  if (path := _layout.reference_image_path(run.out, fov, dataset.reference_image,
                                                           dataset.reference_image_import)) in declared}
        ownership_figures(run.out, assembly.assignment, _layout.stitching_dir(run.out, sample), record.folder,
                          run.inputs(), images)


def report(run):
    """Dataset.report of one sample into report/<sample>/, with the assignment name of report_assignment: the assembly
    block's (None for molecules only), else the assignment block's, which the assign tasks saved per FOV, else None."""
    from starfinder.dataset.workflow import _report_block
    from starfinder.workflow._translate import _base, report_assignment
    configuration = run.configuration.plain()
    config = _report_block(dict(_base(configuration), report=configuration["report"]))
    from starfinder.report._collect import PAGE_FILE
    from starfinder.report._schema import REPORT_FILE, TABLE_DIR
    import shutil
    sample = run.identity["sample"]
    report_dir = _layout.report_dir(run.out, sample)
    if not (report_dir / REPORT_FILE).exists():
        (report_dir / PAGE_FILE).unlink(missing_ok=True)
        tables = report_dir / TABLE_DIR
        if tables.is_dir():
            shutil.rmtree(tables)
    run.dataset().report(sample, list(run.manifest["membership"]["samples"][sample]),
                         assignment=report_assignment(configuration), config=config,
                         output_dir=report_dir)


def _held_report(run):
    """The sample report of an export held for review: the report task's command function through that task's own
    view, once every other task it needs has a succeeded record. It reads the FOVs' records and placement.json (the
    held export left no export.json) and writes report/<sample>/, but no report record, so the report task runs
    after the approval as before. Nothing when the plan has no report task."""
    import logging
    from starfinder.workflow._shape import COMMANDS
    from starfinder.workflow._task import Run, read_record
    from starfinder.workflow._translate import check_api
    logger = logging.getLogger("starfinder.workflow")
    sample = run.identity["sample"]
    task = next((t for t in run.manifest["tasks"] if t["kind"] == "report" and t["identity"] == {"sample": sample}),
                None)
    if task is None:
        logger.info(f"the plan has no report of sample {sample}: no report of the held export")
        return
    waiting = [need for need in task["needs"]
               if need != run.id and (read_record(run.manifest, need) or {}).get("status") != "succeeded"]
    if waiting:
        logger.warning(f"the report of sample {sample} is not made: {task['id']} also needs {', '.join(waiting)}, "
                       "which have no record that says succeeded")
        return
    held = Run(run.manifest_path, task["id"], COMMANDS["report"], run.context)
    check_api(held.configuration, held.api, held.task["blocks"], "report")
    report(held)
    logger.info(f"made the report of the held export: {_layout.report_dir(run.out, sample) / 'report.html'}")


def compare(run):
    """Dataset.compare_samples of every sample whose report is an input, in the membership's order (fovs.csv's; the
    task's declared inputs are sorted), into report/."""
    reported = {path.parent.name for path in run.inputs("report.json")}
    samples = [sample for sample in run.manifest["membership"]["samples"] if sample in reported]
    run.dataset().compare_samples(samples, report_root=_layout.report_root(run.out))


#: The function of each kind.
FUNCTIONS = {"summarize": summarize, "merge": merge, "fov": fov, "tile_global": tile_global, "tile": tile,
             "tile_merge": tile_merge, "morphology": morphology, "segment": segment, "assign": assign,
             "stitch": stitch,
             "bigstitcher_macro": bigstitcher_macro, "bigstitcher_engine": bigstitcher_check,
             "bigstitcher_import": bigstitcher_import, "export": export, "report": report, "compare": compare}
