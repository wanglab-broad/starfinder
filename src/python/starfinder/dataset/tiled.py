"""Tiled execution of one FOV: the tile grid and halo, the global task, the tile task and the merge (D5).

design/specs/orchestration-tiled.md. A FOV that does not fit one job runs as three kinds of task: the global task
(:func:`run_tile_global`) runs ``FOV.run`` up to global registration and the post-registration steps in streaming
mode, writes each registered round once into the chunked checkpoint (``starfinder.io.ChunkedCheckpointWriter``),
the reference image as direct writes it and the FOV-level statistics a window cannot compute; each tile task
(:func:`run_tile`) reads its window of every round, detects with the FOV-level thresholds (``threshold_mode
"absolute"``), extracts, decodes and scores, and keeps the candidates and reads of its core; the merge
(:func:`merge_tiles`) gives them the FOV's identities in the detector's order, deduplicates and filters, and writes
``run.json``, the molecule table and the checkpoint stages under the same contract as direct execution.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from starfinder import _checkpoint, _layout
from starfinder._records import _read_table, _write_table, file_sha256, jsonable, tuples, write_json

__all__ = ["Tile", "detector_order", "largest_tile_size", "merge_tiles", "predicted_tile_memory", "run_tile",
           "run_tile_global", "tile_footprints", "tile_grid", "unbounded_operations"]

#: The formats of the global task's record and of a tile's record.
GLOBAL_FORMAT = "starfinder.tiled.global/1"
TILE_FORMAT = "starfinder.tiled.tile/1"
#: The local registration margin's largest displacement in voxels (the largest cap of the synthetic deformation
#: presets); a proposal of the tiled record (OT1), which the validation calibrates.
LOCAL_REGISTRATION_DISPLACEMENT = 30
#: The resource model of a tile task (the tiled record, "The resource model"): peak bytes ``a + b * W``, ``W`` the
#: bytes of the window of every round; ``a`` and ``b`` bound every measured tile of thick_medium and the real FOV.
TILE_MEMORY_INTERCEPT = 0.21e9
TILE_MEMORY_SLOPE = 1.93
#: The chunk edge whose multiples the tile-size check proposes.
SIZE_STEP = 128


@dataclass(frozen=True)
class Tile:
    """One tile of a FOV's grid: its core and its window, half-open Y and X ranges in FOV indices.

    Parameters
    ----------
    row : int
        Zero-based row of the grid.
    col : int
        Zero-based column of the grid.
    core_yx : tuple of int
        ``(y0, y1, x0, x1)`` of the core; the cores partition the FOV's plane.
    window_yx : tuple of int
        ``(y0, y1, x0, x1)`` of the window: the core grown by the halo on every side, clipped to the FOV.
    """
    row: int
    col: int
    core_yx: tuple[int, int, int, int]
    window_yx: tuple[int, int, int, int]

    @property
    def id(self) -> str:
        """``r<row>c<col>``: the tile's identity, the last part of its task id and the name of its folder."""
        return f"r{self.row}c{self.col}"

    def contains(self, y, x):
        """Whether positions lie in the half-open core (no rounding: floating coordinates are owned as they are).

        Parameters
        ----------
        y : float or numpy.ndarray
            Y positions in FOV indices.
        x : float or numpy.ndarray
            X positions in FOV indices, of the shape of ``y``.

        Returns
        -------
        bool or numpy.ndarray
            True where ``y0 <= y < y1`` and ``x0 <= x < x1`` of the core.
        """
        y0, y1, x0, x1 = self.core_yx
        return (y >= y0) & (y < y1) & (x >= x0) & (x < x1)


def tile_grid(shape_yx, tile_size_yx, halo) -> tuple[Tile, ...]:
    """The tile grid of a FOV: ``ceil(Y / T)`` rows and ``ceil(X / T)`` columns, row-major.

    Parameters
    ----------
    shape_yx : sequence of int
        The FOV's Y and X sizes in voxels.
    tile_size_yx : int
        The largest core edge ``T`` in Y and X.
    halo : int
        The halo in voxels; the window is the core grown by it on every side and clipped to the FOV.

    Returns
    -------
    tuple of Tile
        The tiles, row-major. The core boundaries are ``k * Y // n_y`` and ``k * X // n_x`` (the rule of the retired
        subtile windows), so every core is at most ``T`` and the cores differ by at most one voxel.
    """
    height, width = (int(v) for v in shape_yx)
    size, halo = int(tile_size_yx), int(halo)
    if size < 1 or halo < 0 or height < 1 or width < 1:
        raise ValueError("the tile grid needs positive sizes and a nonnegative halo")
    n_y, n_x = math.ceil(height / size), math.ceil(width / size)
    tiles = []
    for row in range(n_y):
        y0, y1 = row * height // n_y, (row + 1) * height // n_y
        for col in range(n_x):
            x0, x1 = col * width // n_x, (col + 1) * width // n_x
            tiles.append(Tile(row, col, (y0, y1, x0, x1),
                              (max(y0 - halo, 0), min(y1 + halo, height), max(x0 - halo, 0), min(x1 + halo, width))))
    return tuple(tiles)


# --- the footprints and the resource model ---------------------------------------------------------------------

def _detection_configs(spot_finding):
    from starfinder.spot_finding import SpotFindingPlan
    if spot_finding is None:
        return []
    if isinstance(spot_finding, SpotFindingPlan):
        return [spot_finding.config] + [o.config for o in spot_finding.channel_overrides]
    return [spot_finding]


def unbounded_operations(pipeline, local_registration=None) -> list[str]:
    """The operations of a tile without a finite footprint, each named; [] when tiled equals direct.

    Parameters
    ----------
    pipeline : PipelineConfig
        The processing of the FOV.
    local_registration : RegistrationRecipe or None
        The tiles' local registration; it is reported apart, by :func:`tile_footprints`.

    Returns
    -------
    list of str
        One line per operation: a detector other than ``local_maxima`` (no declared footprint or order, OT2),
        ``local_maxima`` with ``min_distance_voxels`` above 1 or a ``merge_radius_zyx`` (greedy over the channel),
        and a detection plan with rounds (whose merged order is not declared).
    """
    from starfinder.spot_finding import LocalMaximaConfig, SpotFindingPlan
    problems = []
    for config in _detection_configs(pipeline.spot_finding):
        if not isinstance(config, LocalMaximaConfig):
            problems.append(f"spot_finding method {config.method!r} declares no footprint and no order of its "
                            "candidates (only local_maxima does)")
            continue
        if config.min_distance_voxels > 1:
            problems.append(f"spot_finding.min_distance_voxels {config.min_distance_voxels}: the spacing of peaks is "
                            "greedy over the whole channel")
        if config.merge_radius_zyx is not None:
            problems.append("spot_finding.merge_radius_zyx: the within-channel merge is greedy over the whole channel")
    if isinstance(pipeline.spot_finding, SpotFindingPlan) and pipeline.spot_finding.rounds is not None:
        problems.append("spot_finding.rounds: the merged order of a detection in several rounds is not declared")
    return problems


def tile_footprints(pipeline, local_registration=None) -> dict:
    """The lateral footprint in voxels of each operation a tile runs; the halo's minimum is their maximum.

    Parameters
    ----------
    pipeline : PipelineConfig
        The processing of the FOV.
    local_registration : RegistrationRecipe or None
        The tiles' local registration, or None.

    Returns
    -------
    dict
        ``spot_finding`` (``local_maxima``: ``min_distance_voxels``), ``reads_extraction`` (the larger of the box's and
        the background ring's Y and X radii) and ``local_registration`` (per step the largest displacement,
        ``LOCAL_REGISTRATION_DISPLACEMENT``, plus for demons ``3 * smoothing_sigma * 2 ** (levels - 1)``), each
        present when the operation runs. Decoding, scoring and filtering act per read (footprint 0); deduplication
        runs in the merge.
    """
    from starfinder.spot_finding import LocalMaximaConfig
    footprints = {}
    detection = [c.min_distance_voxels for c in _detection_configs(pipeline.spot_finding)
                 if isinstance(c, LocalMaximaConfig)]
    if detection:
        footprints["spot_finding"] = int(max(detection))
    if pipeline.extraction is not None:
        radii = list(pipeline.extraction.neighborhood_radius_zyx[1:])
        if pipeline.extraction.background is not None:
            radii += list(pipeline.extraction.background.outer_radius_zyx[1:])
        footprints["reads_extraction"] = int(math.ceil(max(radii)))
    if local_registration is not None:
        margins = []
        for step in local_registration.steps:
            config = step.config
            margin = LOCAL_REGISTRATION_DISPLACEMENT
            if getattr(config, "method", None) == "demons":
                margin += 3 * float(config.smoothing_sigma) * 2 ** (len(config.iterations) - 1)
            margins.append(margin)
        footprints["local_registration"] = int(math.ceil(max(margins)))
    return footprints


def predicted_tile_memory(tile_size_yx, halo, *, z, rounds, channels, bytes_per_voxel) -> float:
    """The resource model's predicted peak of a tile task in bytes: ``a + b * W``.

    Parameters
    ----------
    tile_size_yx : int
        The tile size ``T``.
    halo : int
        The halo ``h``.
    z : int
        The FOV's Z size.
    rounds : int
        The pipeline rounds ``R`` (each with its extraction-source snapshot counted as a round of its own).
    channels : int
        The channels ``C``.
    bytes_per_voxel : int
        The bytes per voxel ``s`` of the chunked checkpoint.

    Returns
    -------
    float
        ``TILE_MEMORY_INTERCEPT + TILE_MEMORY_SLOPE * R * Z * (T + 2h) ** 2 * C * s``.
    """
    window = rounds * z * (int(tile_size_yx) + 2 * int(halo)) ** 2 * channels * bytes_per_voxel
    return TILE_MEMORY_INTERCEPT + TILE_MEMORY_SLOPE * window


def largest_tile_size(memory_bytes, halo, **geometry) -> int | None:
    """The tile size the tile-size check proposes for a task memory.

    Parameters
    ----------
    memory_bytes : float
        The tile task's memory in bytes (``execution.tasks.tile.memory_mb`` times 10**6).
    halo : int
        The halo ``h``.
    **geometry : dict
        ``z``, ``rounds``, ``channels`` and ``bytes_per_voxel``, as for :func:`predicted_tile_memory`.

    Returns
    -------
    int or None
        The largest multiple of 128 whose predicted peak fits ``memory_bytes``; else the largest size that fits; None
        when no size fits (the model's intercept alone exceeds the memory).
    """
    size = None
    for step in (SIZE_STEP, 1):
        top = (memory_bytes - TILE_MEMORY_INTERCEPT) / (TILE_MEMORY_SLOPE * geometry["rounds"] * geometry["z"]
                                                         * geometry["channels"] * geometry["bytes_per_voxel"])
        if top <= 0:
            return None
        edge = int(math.floor(math.sqrt(top))) - 2 * int(halo)
        size = edge // step * step
        if size >= 1:
            return size
    return None


# --- the global task ---------------------------------------------------------------------------------------------

def _plan_configs(fov, spot_finding):
    """(the base config, the SpotFindingPlan without rounds, the effective config of each channel) of the FOV."""
    from starfinder.spot_finding import SpotFindingPlan
    from starfinder.spot_finding import _channel_configs
    single, _ = fov._detection_plan(spot_finding)
    plan = single if isinstance(single, SpotFindingPlan) else SpotFindingPlan(single)
    return plan.config, plan, _channel_configs(plan, len(fov.dataset.channel_order))


def _detection_statistics(fov, spot_finding, image) -> dict:
    """The FOV-level detection statistics of the reference round's detection image: per channel the noise record
    (zero fraction, median, MAD, threshold) of its configured mode, and whether the channel is constant."""
    from starfinder.spot_finding import LocalMaximaConfig
    from starfinder.spot_finding._methods import _noise_record
    _, _, configs = _plan_configs(fov, spot_finding)
    labels = list(fov.dataset.channel_order)
    records = []
    for c, config in enumerate(configs):
        if not isinstance(config, LocalMaximaConfig):
            raise ValueError(f"tiled execution detects with local_maxima only; channel {labels[c]!r} uses "
                             f"{config.method!r}")
        channel = image[..., c]
        record = _noise_record(image, channel, config.threshold_mode, config.threshold_value)
        records.append(dict(record, channel=labels[c], constant=bool(channel.min() == channel.max())))
    return {"channels": records}


def run_tile_global(fov, pipeline, *, execution, directory, checkpoints=None, compression=None) -> dict:
    """The global task of tiled execution for one FOV.

    Parameters
    ----------
    fov : FOV
        The FOV, with its dataset (codebook loaded when the pipeline decodes).
    pipeline : PipelineConfig
        The FOV's whole processing; the global task runs it without detection and readout.
    execution : ExecutionConfig
        The execution of ``FOV.run``; the global task runs in streaming mode whatever its mode.
    directory : str or pathlib.Path
        The global task's folder (``fovs/<fov>/tiled``): it receives ``registered.h5``, ``global.json`` and
        ``run.json``.
    checkpoints : CheckpointConfig or None
        The record settings of the global task's ``run.json`` (its stages are ignored); None hashes the inputs.
    compression : str or None
        The chunked checkpoint's compression: None (uncompressed, the default) or ``"gzip"``.

    Returns
    -------
    dict
        The global record, as written to ``global.json``: ``format``, ``identity``, ``rounds``, ``shape_zyxc``,
        ``dtype``, ``chunk_zyxc``, ``checkpoint`` (path relative to ``directory`` and the entry of each round, its
        SHA-256 among them), ``reference_metadata``, ``statistics`` (the detection thresholds and noise records per
        channel, and the extraction's image background and noise per round and channel) and ``extraction_source``.

    Notes
    -----
    ``FOV.run`` runs load, rotation, the preprocessing recipe's steps, global registration and the post-registration
    steps in streaming mode; each pipeline round is written into the chunked checkpoint as soon as it is processed,
    with its extraction-source snapshot, so that one moving round and the reference round are resident at once. The
    reference image is not written here (the command writes it as direct does, ``FOV.save_reference_image``).
    """
    from starfinder.dataset.config import CheckpointConfig, ExecutionConfig
    from starfinder.io.chunked import ChunkedCheckpointWriter
    from starfinder.readout.extraction import _image_background
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reduced = replace(pipeline, spot_finding=None, extraction=None, decoding=None, scoring=None, deduplication=None,
                      filtering=None)
    recipe = pipeline.preprocessing
    source = recipe.extraction_source if recipe is not None else None
    dataset = fov.dataset
    sequencing = set(dataset.rounds.sequencing_rounds)
    measured = pipeline.extraction is not None and pipeline.extraction.background is not None
    background = {}
    identity = {"dataset_id": dataset.dataset_id, "sample_id": dataset.sample_id, "fov_id": fov.fov_id}
    store = directory / _layout.CHUNKED_CHECKPOINT
    stream = ExecutionConfig("streaming", False, execution.device)
    settings = CheckpointConfig(stages=(), overwrite=True) if checkpoints is None else replace(
        checkpoints, stages=(), overwrite=True, directory=None)
    with ChunkedCheckpointWriter(store, identity=identity, channel_labels=dataset.channel_order,
                                 compression=compression) as writer:
        def sink(name):
            snapshots = {source: fov.snapshots[name][source]} if source is not None else {}
            writer.write_round(name, fov.images[name], fov.metadata[name], snapshots)
            if measured and name in sequencing:
                background[name] = _image_background(snapshots[source] if source is not None else fov.images[name])
        fov._round_sink, fov._record_folder = sink, directory
        try:
            fov.run(reduced, execution=stream, checkpoints=settings)
        finally:
            fov._round_sink, fov._record_folder = None, None
        entries = writer.record
    # FOV.run made the (empty) stage folder of the global run.json; the chunked checkpoint is its registered stage.
    empty = _layout.checkpoint_dir(directory)
    if empty.is_dir() and not any(empty.iterdir()):
        empty.rmdir()
    ref = dataset.rounds.reference_round
    image = fov.images[ref]
    statistics = {"detection": _detection_statistics(fov, pipeline.spot_finding, image)
                  if pipeline.spot_finding is not None else None, "image_background": None}
    if measured:
        rounds = [r for r in dataset.rounds.sequencing_rounds]
        statistics["image_background"] = {
            "round_labels": rounds, "channel_labels": list(dataset.channel_order),
            "background": [[float(background[r][0][c]) for r in rounds] for c in range(len(dataset.channel_order))],
            "noise": [[float(background[r][1][c]) for r in rounds] for c in range(len(dataset.channel_order))]}
    record = {"format": GLOBAL_FORMAT, "identity": identity, "rounds": list(entries),
              "channel_labels": list(dataset.channel_order), "shape_zyxc": list(image.shape),
              "dtype": str(image.dtype), "chunk_zyxc": entries[ref]["chunk_zyxc"],
              "checkpoint": {"path": store.name, "rounds": entries},
              "reference_metadata": jsonable(asdict(fov.metadata[ref])), "extraction_source": source,
              "statistics": statistics}
    write_json(record, directory / _layout.GLOBAL_RECORD)
    return record


# --- the tile task -----------------------------------------------------------------------------------------------

def _absolute_plan(fov, spot_finding, statistics):
    """The detection plan of a tile: each channel's configured config with its FOV-level threshold, absolute."""
    from starfinder.spot_finding import ChannelOverride, SpotFindingPlan
    _, _, configs = _plan_configs(fov, spot_finding)
    channels = statistics["detection"]["channels"]
    absolute = [replace(config, threshold_mode="absolute", threshold_value=float(entry["threshold"]))
                for config, entry in zip(configs, channels)]
    labels = tuple(fov.dataset.channel_order)
    return SpotFindingPlan(absolute[0], tuple(ChannelOverride(label, config)
                                              for label, config in zip(labels[1:], absolute[1:])))


def _subset_intensities(result, keep, ids):
    fields = {name: getattr(result, name)[keep] for name in ("values", "valid", "box_voxels", "background", "noise",
                                                            "background_voxels") if getattr(result, name) is not None}
    return replace(result, spot_ids=ids, **fields)


def run_tile(fov, pipeline, tile, *, store, statistics, directory, halo, local_registration=None,
             post_local_registration=None, execution=None, attempt=1) -> dict:
    """The tile task of tiled execution: one tile's window detected, extracted, decoded and scored.

    Parameters
    ----------
    fov : FOV
        A new FOV of the dataset (codebook loaded when the pipeline decodes); it becomes the tile (``fov.tile``).
    pipeline : PipelineConfig
        The FOV's whole processing; the tile runs its detection, extraction, decoding and scoring.
    tile : Tile
        The tile, from :func:`tile_grid` of the FOV's shape.
    store : str or pathlib.Path
        The chunked checkpoint the global task wrote.
    statistics : dict
        The global record's ``statistics`` (the FOV-level thresholds).
    directory : str or pathlib.Path
        The tile's folder (``fovs/<fov>/tiles/r<row>c<col>``): it receives ``candidates.csv``, ``reads.csv`` and
        ``tile.json``.
    halo : int
        The halo the grid was made with (recorded).
    local_registration : RegistrationRecipe or None
        A registration of each moving round's window on the reference round's window; None registers nothing.
    post_local_registration : tuple of PreprocessingStep or None
        Steps run on every round of the window after the local registration; None runs none.
    execution : ExecutionConfig or None
        The device of the run (the tile always runs in batch mode); None is the CPU.
    attempt : int
        The task's attempt, recorded.

    Returns
    -------
    dict
        The tile record, as written to ``tile.json``.

    Notes
    -----
    The tile detects in the namespace ``[dataset, sample, fov, "r<row>c<col>"]`` with the detector's own IDs, which
    the merge replaces. It keeps the candidates whose (y, x), in FOV coordinates (the window's start added), lie in its
    half-open core, and their reads; ``candidates.csv`` holds the columns of the ``candidates`` stage table and
    ``reads.csv`` those of the ``pre_qc`` stage table (decoding and scoring). Filtering and deduplication run in the
    merge.
    """
    from starfinder.dataset.config import ExecutionConfig
    from starfinder.io.chunked import read_window
    from starfinder.preprocessing import PreprocessingRecipe
    directory = Path(directory)
    fov.tile = tile.id
    source = None
    recipe = pipeline.preprocessing
    if recipe is not None:
        source = recipe.extraction_source
    rounds = fov._pipeline_rounds()
    for name in rounds:
        loaded = read_window(store, name, tile.window_yx)
        metadata = replace(loaded.metadata, frame_id=f"{loaded.metadata.frame_id.rsplit('/window:', 1)[0]}"
                                                     f"/tile:{tile.id}")
        fov.images[name], fov.metadata[name] = loaded.image, metadata
        if source is not None:
            fov.snapshots[name] = {source: read_window(store, name, tile.window_yx, snapshot=source).image}
    if source is not None:
        # The extraction reads the recorded source without a recipe (FOV.run, as after load_checkpoint).
        fov.preprocessing_record = {"recipe": {"extraction_source": source}}
    post = None
    if post_local_registration:
        if source is not None:
            raise ValueError("processing.tiled.post_local_registration with an extraction source is not supported")
        post = PreprocessingRecipe(post_registration=tuple(post_local_registration))
    plan = _absolute_plan(fov, pipeline.spot_finding, statistics)
    run = replace(pipeline, load=None, rotation_degrees=None, preprocessing=post, registration=local_registration,
                  spot_finding=plan, deduplication=None, filtering=None)
    device = "cpu" if execution is None else execution.device
    fov.run(run, execution=ExecutionConfig("batch", False, device), checkpoints=None)
    wy0, _, wx0, _ = tile.window_yx
    spots = fov.spot_result.spots.copy()
    spots["y"] = spots["y"] + wy0
    spots["x"] = spots["x"] + wx0
    keep = tile.contains(spots["y"].to_numpy(), spots["x"].to_numpy())
    core = spots[keep].reset_index(drop=True)
    spot_core = replace(fov.spot_result, spots=core)
    intensity = None
    if fov.intensity_result is not None:
        intensity = _subset_intensities(fov.intensity_result, keep, tuple(core.spot_id))
    frame = _checkpoint.candidates_frame(spot_core, intensity)
    directory.mkdir(parents=True, exist_ok=True)
    candidates, candidate_dtypes = _write_table(frame, directory, "candidates", "csv")
    reads_result = fov.scoring_result or fov.decoding_result
    ids = set(core.spot_id)
    if reads_result is not None:
        reads = reads_result.table[reads_result.table.spot_id.isin(ids)].reset_index(drop=True)
    else:
        reads = pd.DataFrame({"spot_id": pd.array([], dtype="string")})
    reads_name, reads_dtypes = _write_table(reads, directory, "reads", "csv")
    diagnostics = {"spot_finding": _checkpoint._json_diagnostics(fov.spot_result.diagnostics),
                   "extraction": None if fov.intensity_result is None else
                   _checkpoint._json_diagnostics(fov.intensity_result.diagnostics),
                   "decoding": None if fov.decoding_result is None else
                   _checkpoint._json_diagnostics(fov.decoding_result.diagnostics)}
    record = {"format": TILE_FORMAT,
              "identity": {"dataset_id": fov.dataset.dataset_id, "sample_id": fov.dataset.sample_id,
                           "fov_id": fov.fov_id, "tile": tile.id},
              "row": tile.row, "col": tile.col, "core_yx": list(tile.core_yx), "window_yx": list(tile.window_yx),
              "halo": int(halo), "spot_namespace": fov.spot_result.spot_namespace,
              "thresholds": {"configured": [{"channel": e["channel"], "threshold": e["threshold"]}
                                            for e in statistics["detection"]["channels"]],
                             "used": list(fov.spot_result.diagnostics["thresholds"])},
              "local_registration": local_registration is not None,
              "counts": {"candidates_window": int(len(spots)), "candidates_core": int(len(core)),
                         "reads_window": 0 if reads_result is None else int(len(reads_result.table)),
                         "reads_core": int(len(reads))},
              "spot_columns": list(core.columns),
              "tables": {"candidates": {"path": candidates, "dtypes": candidate_dtypes,
                                        "sha256": "sha256:" + file_sha256(directory / candidates)},
                         "reads": {"path": reads_name, "dtypes": reads_dtypes,
                                   "sha256": "sha256:" + file_sha256(directory / reads_name)}},
              "decoded": reads_result is not None, "scored": fov.scoring_result is not None,
              "diagnostics": diagnostics, "attempt": attempt}
    write_json(record, directory / "tile.json")
    return record


# --- the merge ---------------------------------------------------------------------------------------------------

def _namespace(fov):
    return json.dumps([fov.dataset.dataset_id, fov.dataset.sample_id, fov.fov_id, None], separators=(",", ":"))


def _merged_detection(fov, pipeline, statistics, first, spots):
    """The FOV's SpotFindingResult of the merged candidates, with direct's diagnostics from the FOV-level statistics."""
    from starfinder.spot_finding import SpotFindingResult
    from starfinder.spot_finding import _noise_warning
    base, plan, configs = _plan_configs(fov, pipeline.spot_finding)
    labels = tuple(fov.dataset.channel_order)
    channels = statistics["detection"]["channels"]
    diagnostics = {key: tuples(value) for key, value in first.items()
                   if key not in ("thresholds", "counts", "outcomes", "noise", "effective_settings", "warnings")}
    diagnostics["thresholds"] = tuple(float(e["threshold"]) for e in channels)
    counts = spots["channel"].value_counts() if "channel" in spots else pd.Series(dtype=int)
    diagnostics["counts"] = {labels[c]: int(counts.get(c, 0)) for c in range(len(labels))}
    diagnostics["outcomes"] = {labels[c]: "constant" if channels[c]["constant"] else
                               "ok" if diagnostics["counts"][labels[c]] else "empty" for c in range(len(labels))}
    noise = {labels[c]: {k: channels[c][k] for k in ("zero_fraction", "median", "mad", "threshold")}
             for c in range(len(labels))}
    diagnostics["noise"] = noise
    diagnostics["effective_settings"] = {labels[c]: tuples(jsonable(configs[c])) for c in range(len(labels))}
    diagnostics["warnings"] = tuple(_noise_warning(noise[label], label, None) for label in labels
                                    if noise[label]["mad"] == 0 or noise[label]["zero_fraction"] > 0.5)
    from starfinder.image import ImageMetadata
    metadata = ImageMetadata(**tuples(statistics["reference_metadata"]))
    return SpotFindingResult(spots, metadata, _namespace(fov), base, diagnostics, plan)


def detector_order(spots, round_labels=None):
    """The candidates sorted in the detector's order, which numbers direct's spot IDs.

    Parameters
    ----------
    spots : pandas.DataFrame
        Candidates with z, y, x, and for ``local_maxima`` channel and peak_intensity; a ``round`` column for a plan
        with rounds.
    round_labels : sequence of str or None
        The detection rounds in FOV.run order, for a ``round`` column.

    Returns
    -------
    pandas.DataFrame
        The rows of ``spots`` in order (index reset): by round (FOV.run order), then for ``local_maxima`` by channel,
        decreasing peak value, then z, y, x (``_maxima`` and ``peak_local_max``: a stable sort of C-order peaks by
        decreasing value), else by channel, then z, y, x.
    """
    frame = spots.copy()
    keys, ascending = [], []
    if "round" in frame and round_labels is not None:
        frame["_round"] = frame["round"].map({r: i for i, r in enumerate(round_labels)})
        keys.append("_round"), ascending.append(True)
    if "channel" in frame:
        keys.append("channel"), ascending.append(True)
    if "peak_intensity" in frame:
        keys.append("peak_intensity"), ascending.append(False)
    keys += ["z", "y", "x"]
    ascending += [True, True, True]
    frame = frame.sort_values(keys, ascending=ascending, kind="stable").reset_index(drop=True)
    return frame.drop(columns=[c for c in ("_round",) if c in frame])


def merge_tiles(fov, pipeline, tiles, *, tile_directories, global_record, global_run_record, execution, checkpoints,
                tiled=None) -> dict:
    """The merge of tiled execution: the tiles' core candidates and reads as the FOV's, then deduplication, filtering
    and the FOV's records, as direct writes them.

    Parameters
    ----------
    fov : FOV
        A new FOV of the dataset (codebook loaded when the pipeline decodes).
    pipeline : PipelineConfig
        The FOV's whole processing.
    tiles : sequence of Tile
        Every tile of the grid; each must have reported.
    tile_directories : mapping
        Tile id to the tile's folder.
    global_record : dict
        The global task's ``global.json``.
    global_run_record : dict
        The global task's ``run.json``: its inputs, steps, preprocessing and registration records go into the FOV's.
    execution : ExecutionConfig
        The execution recorded in the FOV's ``run.json`` (direct's).
    checkpoints : CheckpointConfig
        The stages to write (``candidates`` and ``pre_qc``; ``registered`` is the chunked checkpoint) and the record
        settings; the FOV folder is ``fovs/<fov>``.
    tiled : dict or None
        The ``tiled`` section added to ``run.json`` (the grid, the checkpoint, the tiles, the statistics).

    Returns
    -------
    dict
        ``candidates`` and ``reads`` (the merged counts) and ``run_record`` (the path of the FOV's ``run.json``).

    Notes
    -----
    Each candidate belongs to the one tile whose core holds it; a candidate at one (z, y, x, channel) in two tiles,
    or a tile without its files, raises. The merged candidates are sorted in the detector's order
    (:func:`detector_order`) and numbered from 0 in the FOV's namespace ``[dataset, sample, fov, null]``, which is
    direct's numbering; each read takes its candidate's ID. The candidates' image background is the global task's.
    """
    from starfinder.readout import IntensityExtractionResult, ReadDecodingResult, ReadScoringResult
    from starfinder.readout.scoring import SCORE_COLUMNS
    from starfinder._records import tuples as _tuples
    statistics = dict(global_record["statistics"], reference_metadata=global_record["reference_metadata"])
    frames, reads, records = [], [], {}
    for tile in tiles:
        folder = Path(tile_directories[tile.id])
        path = folder / "tile.json"
        if not path.is_file():
            raise ValueError(f"tile {tile.id} reported no tile.json in {folder}")
        record = json.loads(path.read_text())
        if record["core_yx"] != list(tile.core_yx):
            raise ValueError(f"tile {tile.id}: its core {record['core_yx']} is not the grid's {list(tile.core_yx)}")
        records[tile.id] = record
        tables = record["tables"]
        frame = _read_table(folder / tables["candidates"]["path"], tables["candidates"]["dtypes"])
        frames.append(frame.assign(_tile=tile.id))
        read = _read_table(folder / tables["reads"]["path"], tables["reads"]["dtypes"])
        reads.append(read.assign(_tile=tile.id))
    candidates = pd.concat(frames, ignore_index=True)
    key = [c for c in ("z", "y", "x", "channel", "round") if c in candidates]
    if candidates.duplicated(key).any():
        raise ValueError("a candidate is owned by two tiles: " + candidates[candidates.duplicated(key, keep=False)]
                         [key + ["_tile"]].head().to_string(index=False))
    detection_rounds = fov._detection_order(pipeline.spot_finding.rounds) if getattr(
        pipeline.spot_finding, "rounds", None) else None
    candidates = detector_order(candidates, detection_rounds)
    namespace = _namespace(fov)
    ids = pd.array([str(i) for i in range(len(candidates))], dtype="string")
    mapping = dict(zip(zip(candidates["_tile"], candidates["spot_id"]), ids))
    candidates["spot_id"] = ids
    candidates["spot_namespace"] = pd.array([namespace] * len(candidates), dtype="string")
    candidates = candidates.drop(columns=["_tile"])
    first = records[tiles[0].id]
    spot_columns = first["spot_columns"]
    dataset = fov.dataset
    channels = tuple(dataset.channel_order)
    rounds = tuple(dataset.rounds.sequencing_rounds)
    if first["diagnostics"]["extraction"] is not None:
        spots, values, valid = _checkpoint.parse_candidates(candidates, rounds, channels)
        background = _checkpoint.parse_background(candidates, rounds, channels)
    else:
        spots, values, valid, background = candidates, None, None, None
    spots = spots[spot_columns].reset_index(drop=True)
    fov.spot_result = _merged_detection(fov, pipeline, statistics, first["diagnostics"]["spot_finding"], spots)
    shape = list(global_record["shape_zyxc"][:3])
    if values is not None:
        measured = {}
        image = global_record["statistics"]["image_background"]
        if background is not None and image is not None:
            bg, noise, bgvox, boxvox = background
            measured = dict(box_voxels=boxvox, background=bg, noise=noise, background_voxels=bgvox,
                            image_background=np.asarray(image["background"], dtype=np.float64),
                            image_noise=np.asarray(image["noise"], dtype=np.float64))
        diagnostics = _tuples(first["diagnostics"]["extraction"])
        diagnostics = {"rounds": {r: dict(entry, source_shape_zyx=tuple(shape))
                                  for r, entry in diagnostics.get("rounds", {}).items()}}
        fov.intensity_result = IntensityExtractionResult(values, tuple(spots.spot_id), namespace, channels, rounds,
                                                         fov.spot_result.metadata, pipeline.extraction, valid,
                                                         diagnostics, **measured)
    if first["decoded"]:
        table = pd.concat(reads, ignore_index=True)
        table["spot_id"] = pd.array([mapping[(t, s)] for t, s in zip(table["_tile"], table["spot_id"])],
                                    dtype="string")
        table["spot_namespace"] = pd.array([namespace] * len(table), dtype="string")
        table = table.drop(columns=["_tile"])
        table = table.assign(_order=table["spot_id"].astype(np.int64)).sort_values("_order", kind="stable")
        table = table.drop(columns=["_order"]).reset_index(drop=True)
        diagnostics = dict(_tuples(first["diagnostics"]["decoding"]))
        if "clipped_count" in diagnostics and values is not None:
            negative = values[values < 0]
            diagnostics["clipped_count"] = int(len(negative))
            diagnostics["clipped_range"] = ((float(negative.min()), float(negative.max())) if len(negative)
                                            else None)
        decoded = table.drop(columns=[c for c in SCORE_COLUMNS if c in table]) if first["scored"] else table
        fov.decoding_result = ReadDecodingResult(decoded, namespace, channels, rounds, pipeline.decoding, diagnostics,
                                                 dataset.readout_mode)
        if first["scored"]:
            reasons = table.qc_reason
            counts = {"total": len(table), "scored": int(reasons.eq("").sum()),
                      "no_assignment": int(reasons.eq("no_assignment").sum()),
                      "background_unavailable": int(reasons.eq("background_unavailable").sum())}
            fov.scoring_result = ReadScoringResult(table, namespace, channels, rounds, pipeline.scoring, counts,
                                                   dataset.readout_mode)
    stages = tuple(s for s in checkpoints.stages if s != "registered")
    checkpoints = replace(checkpoints, stages=stages)
    record = fov._start_run_record(checkpoints, pipeline, execution)
    # The FOV's record holds the global task's inputs and steps, and its preprocessing and registration records
    # where the merge has none (the merged FOV holds no image and no registration result).
    record.data.update(inputs=list(global_run_record.get("inputs") or []),
                       steps=list(global_run_record.get("steps") or []))
    record.kept = {key: global_run_record.get(key) for key in ("preprocessing", "registration", "transforms")}
    fov._run_record = record
    fov._record_path = None
    try:
        if "candidates" in stages and fov.spot_result is not None:
            fov._write_checkpoint(stage="candidates", directory=record.checkpoint_directory,
                                  table_format=checkpoints.table_format)
        if pipeline.deduplication is not None:
            fov.deduplicate_reads(config=pipeline.deduplication)
        if "pre_qc" in stages and fov.decoding_result is not None:
            fov._write_checkpoint(stage="pre_qc", directory=record.checkpoint_directory,
                                  table_format=checkpoints.table_format)
        if pipeline.filtering is not None:
            fov.filter_reads(config=pipeline.filtering)
        if tiled is not None:
            record.data["tiled"] = tiled
        record.finish("succeeded")
        fov._record_path = record.path
    except BaseException as error:
        if record.data["status"] == "running":
            record.finish("failed" if isinstance(error, Exception) else "interrupted", error)
        raise
    finally:
        fov._run_record = None
    return {"candidates": len(spots), "reads": 0 if fov.decoding_result is None else len(fov.decoding_result.table),
            "run_record": record.path}
