"""The API entry of each task kind's command (design/specs/orchestration.md, the table of the commands).

Each function takes the running task (``_task.Run``), reads the resolved configuration through the kind's view,
translates its block with the kept translators and calls the API entry; ``_task.execute`` does the checks, the log,
completion and the records around it. The morphology, segment and assign commands land with W-736 and the tiled
kinds with W-735.
"""
from __future__ import annotations

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
    """(adapted WorkflowConfig with the task's dataset, pipeline with the task's supplied file, stages)."""
    translated = _translate.processing(run.configuration.plain())
    adapted = translated.workflow
    run.dataset(adapted.dataset)
    pipeline = adapted.pipeline
    recipe = pipeline.preprocessing
    if recipe is not None and recipe.supplied_statistics is not None:
        pipeline = replace(pipeline, preprocessing=replace(
            recipe, supplied_statistics=_supplied_file(run, recipe.supplied_statistics)))
    return adapted, pipeline, translated.stages


def _file(run, name):
    """A supplied file of the manifest's inputs (fovs.csv, the codebook, the position file, ...) by its name."""
    path = Path(run.manifest["inputs"]["files"][name]["path"])
    return path if path.is_absolute() else run.root / path


def _codebook(run):
    return _file(run, "codebook")


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


def _from_plain(value):
    """A plain value of the resolved configuration back as the dataclass its ``type`` names (the encodings and
    layouts that cross_block hands the export)."""
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
    """Dataset.export of one sample into export/<sample>/; the molecule routes load the codebook for its genes."""
    config, assembly = _translate.export(run.configuration.plain())
    dataset = run.dataset()
    export_api = run.api["export"]
    if export_api["previews"]:
        raise NotImplementedError("export.previews: the ownership previews of the export command land with the "
                                  "assign command (W-736)")
    if export_api["codebook"] is not None:
        if dataset.readout_mode == "direct":
            dataset.load_direct_panel(_codebook(run))
        else:
            dataset.load_codebook(_codebook(run), encoding=_from_plain(export_api["codebook"]["encoding"]),
                                  layout=_from_plain(export_api["codebook"]["layout"]))
    sample = run.identity["sample"]
    dataset.export(sample, placement=_layout.stitching_dir(run.out, sample), assignment=assembly.assignment,
                   config=config, output_dir=_layout.export_dir(run.out, sample))


def report(run):
    """Dataset.report of one sample into report/<sample>/, with the assignment name the plan resolved."""
    from starfinder.dataset.workflow import _report_block
    from starfinder.workflow._translate import _base
    configuration = run.configuration.plain()
    config = _report_block(dict(_base(configuration), report=configuration["report"]))
    sample = run.identity["sample"]
    run.dataset().report(sample, list(run.manifest["membership"]["samples"][sample]),
                         assignment=run.api["report"]["assignment"], config=config,
                         output_dir=_layout.report_dir(run.out, sample))


def compare(run):
    """Dataset.compare_samples of every sample whose report is an input, in the inputs' order, into report/."""
    samples = [path.parent.name for path in run.inputs("report.json")]
    run.dataset().compare_samples(samples, report_root=_layout.report_root(run.out))


#: The function of each kind; morphology, segment and assign land with W-736, the tiled kinds with W-735.
FUNCTIONS = {"summarize": summarize, "merge": merge, "fov": fov, "stitch": stitch,
             "bigstitcher_macro": bigstitcher_macro, "bigstitcher_engine": bigstitcher_check,
             "bigstitcher_import": bigstitcher_import, "export": export, "report": report, "compare": compare}
