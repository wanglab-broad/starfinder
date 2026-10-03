"""Per-FOV pipeline checkpoints: registered images, candidates and pre-QC reads.

Each stage has a small JSON header (identity, labels, typed configs and the
column dtype map) next to its data. Tables are CSV by default or Parquet with
the ``checkpoint`` extra. Every file is written to a temporary name in its
directory and moved into place with ``os.replace``.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

FORMAT_VERSION = 2
# Header versions the reader accepts: version 1 has the pre-recipe registered layout.
READ_VERSIONS = (1, 2)
STAGES = ("registered", "candidates", "pre_qc")
TABLE_FORMATS = ("csv", "parquet")
NA_TOKEN = "<NA>"
ESCAPE = "\\"
_CONTROL = "[\x00-\x1f\x7f]"
# In-memory dtype -> dtype used to parse CSV text; the header restores the former.
_CSV_DTYPES = {"string": "string", "str": "string", "float64": "float64",
               "int64": "Int64", "Int64": "Int64", "bool": "boolean", "boolean": "boolean"}


def _check_stage(stage):
    if stage not in STAGES:
        raise ValueError(f"unknown checkpoint stage {stage!r}; expected one of {STAGES}")


def _require_parquet():
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise ImportError("Parquet checkpoints require pyarrow; install the 'checkpoint' extra "
                          "or use table_format='csv'") from error


@contextmanager
def _atomic(path):
    """Yield a temporary sibling path; replace ``path`` with it only on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _jsonable(value):
    """Plain JSON values; tuples become lists, types qualified names, NaN/inf null."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def write_json(data, path):
    """Atomically write indented, strict JSON (non-finite floats become null)."""
    with _atomic(path) as tmp:
        tmp.write_text(json.dumps(_jsonable(data), indent=2, allow_nan=False) + "\n")


def _tuples(value):
    """Invert JSON list conversion for results whose sequences are tuples."""
    if isinstance(value, list):
        return tuple(_tuples(v) for v in value)
    if isinstance(value, dict):
        return {k: _tuples(v) for k, v in value.items()}
    return value


def _json_diagnostics(diagnostics):
    """Keep JSON-representable diagnostics; arrays and tables are not saved."""
    kept = {}
    for key, value in diagnostics.items():
        if isinstance(value, (np.ndarray, pd.DataFrame, pd.Series)):
            continue
        kept[key] = _json_diagnostics(value) if isinstance(value, dict) else value
    return kept


def _config(data, classes):
    """Rebuild a frozen config tagged by its ``method`` (or a single class)."""
    data = dict(data)
    method = data.pop("method", None)
    cls = classes[method] if isinstance(classes, dict) else classes
    return cls(**{k: _tuples(v) for k, v in data.items()})


def _metadata(data):
    from starfinder.image import ImageMetadata
    return ImageMetadata(**data)


# --- Tables ---------------------------------------------------------------------

def _dtype_map(frame):
    dtypes = {}
    for column, dtype in frame.dtypes.items():
        name = str(dtype)
        if name not in _CSV_DTYPES:
            raise ValueError(f"checkpoint column {column!r} has unsupported dtype {name}")
        dtypes[str(column)] = name
    return dtypes


def _escape_strings(column):
    """CSV text for a string column: missing values stay NA (written as NA_TOKEN).

    Literal values equal to NA_TOKEN or starting with ESCAPE gain one leading
    ESCAPE, so printable text, including "", "NA" or "<NA>", reads back exactly.
    """
    literal = column.eq(NA_TOKEN).fillna(False) | column.str.startswith(ESCAPE).fillna(False)
    return column.mask(literal, ESCAPE + column)


def _unescape_strings(column):
    missing = column.eq(NA_TOKEN)
    column = column.mask(column.str.startswith(ESCAPE), column.str.slice(len(ESCAPE)))
    return column.mask(missing, pd.NA)


def _write_table(frame, directory, name, table_format):
    """Write a table with its dtype map; return (file name, dtype map)."""
    dtypes = _dtype_map(frame)
    path = Path(directory) / f"{name}.{table_format}"
    if table_format == "parquet":
        _require_parquet()
        with _atomic(path) as tmp:
            frame.to_parquet(tmp, index=False, engine="pyarrow")
    elif table_format == "csv":
        text = frame.copy()
        for column, dtype in dtypes.items():
            if dtype in ("string", "str"):
                strings = frame[column].astype("string")
                # CSV holds printable text only; never write a changed value.
                if strings.str.contains(_CONTROL, regex=True, na=False).any():
                    raise ValueError(f"CSV checkpoint column {column!r} contains a control character "
                                     "(U+0000-U+001F or U+007F); use table_format='parquet' for such data")
                text[column] = _escape_strings(strings)
        with _atomic(path) as tmp:
            text.to_csv(tmp, index=False, float_format="%.17g", na_rep=NA_TOKEN)
    else:
        raise ValueError(f"table_format must be one of {TABLE_FORMATS}")
    return path.name, dtypes


def _read_table(path, dtypes):
    """Read a table and restore the recorded dtype of every column, in order."""
    path = Path(path)
    if path.suffix == ".parquet":
        _require_parquet()
        frame = pd.read_parquet(path, engine="pyarrow")
    else:
        # String columns are read verbatim (no NA parsing), then unescaped.
        strings = [c for c, d in dtypes.items() if d in ("string", "str")]
        frame = pd.read_csv(path, dtype={c: _CSV_DTYPES[d] for c, d in dtypes.items()},
                            keep_default_na=False, float_precision="round_trip",
                            na_values={c: [NA_TOKEN] for c in dtypes if c not in strings})
        for column in strings:
            if column in frame:
                frame[column] = _unescape_strings(frame[column])
    if list(frame.columns) != list(dtypes):
        raise ValueError(f"{path.name}: columns differ from the checkpoint header")
    return frame.astype(dtypes).reset_index(drop=True)


def _signal_columns(round_labels, channel_labels):
    signals = [f"sig_{r}_{c}" for r in round_labels for c in channel_labels]
    valid = [f"valid_{r}" for r in round_labels]
    return signals, valid


def _background_columns(round_labels, channel_labels):
    """bg_<round>_<channel>, noise_<round>_<channel>, bgvox_<round> and boxvox_<round> (option C1)."""
    return ([f"bg_{r}_{c}" for r in round_labels for c in channel_labels],
            [f"noise_{r}_{c}" for r in round_labels for c in channel_labels],
            [f"bgvox_{r}" for r in round_labels], [f"boxvox_{r}" for r in round_labels])


def _wide(array, n):
    """(N, C, R) -> (N, R*C), round-major as the signal columns."""
    return array.transpose(0, 2, 1).reshape(n, array.shape[1] * array.shape[2])


def candidates_frame(spot_result, intensity_result=None):
    """Wide table: identity, spot columns, then sig_<round>_<channel> and valid_<round>.

    Signals are round-major in round-label order, channels in channel order.
    With background measurements (§2.8, option C1) the table continues with
    bg_<round>_<channel> and noise_<round>_<channel> (float64, the same order),
    bgvox_<round> (ring voxels) and boxvox_<round> (box voxels, int64).
    Built with array reshapes; no per-row work.
    """
    spots = spot_result.spots
    frame = spots.copy()
    if "spot_namespace" not in frame:
        frame.insert(0, "spot_namespace", pd.Series(spot_result.spot_namespace, index=frame.index, dtype="string"))
    frame = frame[["spot_namespace", "spot_id"] + [c for c in frame if c not in ("spot_namespace", "spot_id")]]
    if intensity_result is None:
        return frame.reset_index(drop=True)
    if (intensity_result.spot_namespace != spot_result.spot_namespace
            or intensity_result.spot_ids != tuple(spots.spot_id)):
        raise ValueError("intensity identities must match the spot table in order")
    rounds, channels = intensity_result.round_labels, intensity_result.channel_labels
    signals, valid = _signal_columns(rounds, channels)
    measured = intensity_result.background is not None
    background = _background_columns(rounds, channels) if measured else ([], [], [], [])
    names = signals + valid + [c for group in background for c in group]
    if len(set(names) | set(frame.columns)) != len(names) + frame.shape[1]:
        raise ValueError("round/channel labels produce duplicate candidate column names")
    n = len(spots)
    values = _wide(intensity_result.values, n)
    parts = [frame.reset_index(drop=True),
             pd.DataFrame(values, columns=signals, dtype="float64"),
             pd.DataFrame(intensity_result.valid, columns=valid, dtype=bool)]
    if measured:
        bg, noise, bgvox, boxvox = background
        parts += [pd.DataFrame(_wide(intensity_result.background, n), columns=bg, dtype="float64"),
                  pd.DataFrame(_wide(intensity_result.noise, n), columns=noise, dtype="float64"),
                  pd.DataFrame(intensity_result.background_voxels, columns=bgvox, dtype="int64"),
                  pd.DataFrame(intensity_result.box_voxels, columns=boxvox, dtype="int64")]
    return pd.concat(parts, axis=1)


def parse_candidates(frame, round_labels, channel_labels):
    """Split a wide table into (spot table, values N×C×R, valid N×R) without row loops."""
    signals, valid = _signal_columns(round_labels, channel_labels)
    missing = [c for c in signals + valid if c not in frame]
    if missing:
        raise ValueError(f"candidate table lacks signal columns {missing[:3]}")
    n = len(frame)
    values = (frame[signals].to_numpy(dtype=np.float64)
              .reshape(n, len(round_labels), len(channel_labels)).transpose(0, 2, 1).copy())
    return frame.drop(columns=signals + valid), values, frame[valid].to_numpy(dtype=bool)


def parse_background(frame, round_labels, channel_labels):
    """(background N×C×R, noise N×C×R, background_voxels N×R, box_voxels N×R) of a wide table.

    None when the table has no background columns (written without them, or before §2.8).
    """
    bg, noise, bgvox, boxvox = _background_columns(round_labels, channel_labels)
    if not any(c in frame for c in bg + noise + bgvox + boxvox):
        return None
    missing = [c for c in bg + noise + bgvox + boxvox if c not in frame]
    if missing:
        raise ValueError(f"candidate table lacks background columns {missing[:3]}")
    n, shape = len(frame), (len(frame), len(round_labels), len(channel_labels))

    def cube(columns):
        return frame[columns].to_numpy(dtype=np.float64).reshape(shape).transpose(0, 2, 1).copy()

    return (cube(bg), cube(noise), frame[bgvox].to_numpy(dtype=np.int64).reshape(n, len(round_labels)),
            frame[boxvox].to_numpy(dtype=np.int64).reshape(n, len(round_labels)))


def _per_round_channel(array, round_labels, channel_labels):
    """C×R array -> {round: {channel: value}} for a JSON header."""
    return {r: {c: array[i, j] for i, c in enumerate(channel_labels)} for j, r in enumerate(round_labels)}


def _from_round_channel(data, round_labels, channel_labels):
    """{round: {channel: value or null}} -> C×R float64 (null is NaN)."""
    return np.array([[np.nan if data[r][c] is None else data[r][c] for r in round_labels] for c in channel_labels],
                    dtype=np.float64).reshape(len(channel_labels), len(round_labels))


# --- Headers --------------------------------------------------------------------

def header_path(directory, stage):
    """Path of a stage's JSON header inside a per-FOV checkpoint directory."""
    _check_stage(stage)
    directory = Path(directory)
    return directory / "registered" / "transforms.json" if stage == "registered" else directory / f"{stage}.json"


def clear_stages(directory):
    """Remove every checkpoint file of a per-FOV directory, headers first.

    Only files this module writes are removed (stage headers and tables,
    registered TIFFs, current ``.ome.tif`` or earlier ``.tif``, registered
    snapshot TIFFs in ``registered/<snapshot>/`` and dense fields); other
    files are left untouched. Removing headers first means an
    interrupted clear never leaves a loadable stale stage.
    """
    directory = Path(directory)
    registered = directory / "registered"
    paths = [header_path(directory, stage) for stage in STAGES]
    paths += [directory / f"{stage}.{fmt}" for stage in STAGES[1:] for fmt in TABLE_FORMATS]
    snapshots = []
    if registered.is_dir():
        paths += sorted(registered.glob("*.tif")) + sorted(registered.glob("*_field.npz"))
        snapshots = sorted(path for path in registered.iterdir() if path.is_dir() and not path.is_symlink())
        paths += [path for directory in snapshots for path in sorted(directory.glob("*.ome.tif"))]
    for path in paths:
        path.unlink(missing_ok=True)
    for path in [*snapshots, registered]:
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def readout_mode(header):
    """The readout mode of a candidates or pre_qc header; a header without the key (before §2.8) is multiplexed."""
    return header.get("readout_mode", "multiplexed")


def read_header(directory, stage):
    """Load a stage header; missing checkpoints raise FileNotFoundError."""
    path = header_path(directory, stage)
    if not path.is_file():
        raise FileNotFoundError(f"no {stage} checkpoint at {path}")
    header = json.loads(path.read_text())
    if header.get("format_version") not in READ_VERSIONS or header.get("stage") != stage:
        raise ValueError(f"{path} is not a version {' or '.join(map(str, READ_VERSIONS))} {stage} checkpoint")
    return header


# --- Registered stage ------------------------------------------------------------

def registered_image_path(directory, round_name, snapshot=None):
    """Path of a registered round image; checkpoints before OME-TIFF used <round>.tif.

    A snapshot of the round is stored as registered/<snapshot>/<round>.ome.tif.
    """
    directory = Path(directory) / "registered"
    return (directory if snapshot is None else directory / snapshot) / f"{round_name}.ome.tif"


def _registered_image(directory, round_name):
    """The current image of a saved round, falling back to the earlier <round>.tif name."""
    path = registered_image_path(directory, round_name)
    earlier = Path(directory) / "registered" / f"{round_name}.tif"
    return earlier if not path.is_file() and earlier.is_file() else path


def write_registered_round(directory, round_name, image, metadata, snapshot=None):
    """Write one registered ZYXC round (or its snapshot) as OME-TIFF with its geometry."""
    from starfinder.io.tiff import save_volume
    if np.asarray(image).ndim != 4:
        raise ValueError(f"registered checkpoint requires ZYXC images; {round_name} is not")
    for label in (round_name, snapshot):
        if label is not None and (not label or Path(label).name != label or label in (".", "..")):
            raise ValueError(f"label {label!r} is not a plain file name")
    path = registered_image_path(directory, round_name, snapshot)
    with _atomic(path) as tmp:
        save_volume(image, tmp, metadata=metadata)
    return path


def _transform_json(result, index, field_name, application_config=None):
    """One step result of transforms.json: the step index, the typed transform and its diagnostics.

    application_config is kept per result only for sequential (version-1) semantics.
    """
    from starfinder.registration._chain import transform_kind
    transform = result.transform
    data = {name: getattr(transform, name) for name in
            ("reference_shape_zyx", "moving_shape_zyx", "reference_metadata", "moving_metadata", "direction", "units")}
    kind = transform_kind(transform)
    data.update(kind=kind)
    if kind == "translation":
        data.update(displacement_zyx=transform.displacement_zyx)
    elif kind == "affine":
        data.update(matrix_zyx=transform.matrix_zyx.tolist(), physical=transform.physical)
    elif kind == "bspline":
        data.update(bspline={name: getattr(transform, name) for name in _BSPLINE_FIELDS}, coefficients=field_name)
    else:
        data.update(field=field_name)
    entry = dict(step=index, transform=data, diagnostics=result.diagnostics)
    if application_config is not None:
        entry.update(application_config=application_config)
    return entry


# Inline B-spline parameters; the coefficients are stored in the round's field file.
_BSPLINE_FIELDS = ("grid_size_xyz", "grid_origin_xyz", "grid_spacing_xyz", "grid_direction_xyz", "order", "spacing_zyx")


def write_registered_header(directory, header, registration_results, registration_record=None):
    """Write transforms.json and a <round>_field.npz per round with dense fields or B-spline coefficients.

    registration_record (FOV.registration_record) gives the semantics
    (``recipe``: one application entry per round with its WarpConfig;
    ``sequential``: a per-result application_config, as read from version 1)
    and the recipe summary. Without an application in the record, a round's
    application is its last result's application_config.
    """
    from starfinder.registration import BSplineTransform, DenseDisplacementTransform
    record = registration_record or {}
    semantics = record.get("semantics", "recipe")
    applications = record.get("application", {})
    directory = Path(directory) / "registered"
    transforms, files = {}, []
    for name, results in registration_results.items():
        arrays = {}
        for i, r in enumerate(results):
            if isinstance(r.transform, DenseDisplacementTransform):
                arrays[f"result_{i}"] = r.transform.displacement_zyx
            elif isinstance(r.transform, BSplineTransform):
                arrays[f"result_{i}"] = r.transform.coefficients
        field_name = f"{name}_field.npz"
        if arrays:
            with _atomic(directory / field_name) as tmp:
                with open(tmp, "wb") as handle:
                    np.savez(handle, **arrays)
            files.append(field_name)
        transforms[name] = [_transform_json(r, i, field_name, r.application_config if semantics == "sequential" else None)
                            for i, r in enumerate(results)]
    header = dict(header, stage="registered", format_version=FORMAT_VERSION, transforms=transforms,
                  registration_semantics=semantics, registration_recipe=record.get("recipe"),
                  applications={} if semantics == "sequential" else
                  {name: {"application_config": applications.get(name, results[-1].application_config)}
                   for name, results in registration_results.items() if results})
    write_json(header, directory / "transforms.json")
    return files + ["transforms.json"]


def _registration_results(directory, header):
    """Rebuild the results, the TransformChain per round and the registration record of a registered header.

    Version 1 checkpoints hold translation and dense results with a
    per-result application_config; they load with semantics ``sequential``
    and without chains. Version 2 rebuilds every kind, a chain per round and
    the round's application WarpConfig (also set on each result).
    """
    from starfinder._registry import config_type_for, names
    from starfinder.registration import (REGISTRATION_METHODS, AffineTransform, BSplineTransform,
        DenseDisplacementTransform, RegistrationDiagnostics, RegistrationResult, TransformChain,
        TranslationTransform, WarpConfig)
    # Saved method values equal the spec names (the discriminator rule).
    methods = {name: config_type_for(REGISTRATION_METHODS, name, "registration method")
               for name in names(REGISTRATION_METHODS)}
    semantics = "sequential" if header["format_version"] == 1 else header.get("registration_semantics", "recipe")
    applications = {name: _config(entry["application_config"], WarpConfig)
                    for name, entry in header.get("applications", {}).items()}
    results, chains = {}, {}
    for name, entries in header["transforms"].items():
        arrays = None
        restored = []
        for i, entry in enumerate(entries):
            data = dict(entry["transform"])
            kind = data.pop("kind")
            data.update(reference_metadata=_metadata(data["reference_metadata"]),
                        moving_metadata=_metadata(data["moving_metadata"]),
                        reference_shape_zyx=tuple(data["reference_shape_zyx"]),
                        moving_shape_zyx=tuple(data["moving_shape_zyx"]))
            if kind in ("dense", "bspline"):
                file_name = data.pop("field" if kind == "dense" else "coefficients")
                if arrays is None:
                    with np.load(Path(directory) / "registered" / file_name) as npz:
                        arrays = {k: npz[k] for k in npz.files}
            if kind == "translation":
                if header["format_version"] == 1:
                    # Version 1 stores the correction c of the pull p - c; the displacement is -c.
                    data.update(displacement_zyx=tuple(-v for v in data.pop("correction_zyx")),
                                direction="reference_to_moving")
                transform = TranslationTransform(**data)
            elif kind == "affine" and header["format_version"] != 1:
                transform = AffineTransform(**data)
            elif kind == "bspline" and header["format_version"] != 1:
                transform = BSplineTransform(**{k: _tuples(v) for k, v in data.pop("bspline").items()},
                                             coefficients=arrays[f"result_{i}"], **data)
            elif kind == "dense":
                transform = DenseDisplacementTransform(displacement_zyx=arrays[f"result_{i}"], **data)
            else:
                raise ValueError(f"unknown transform kind {kind!r} in a version {header['format_version']} checkpoint")
            diagnostics = dict(entry["diagnostics"])
            diagnostics["effective_config"] = _config(diagnostics["effective_config"], methods)
            diagnostics["warnings"] = tuple(diagnostics["warnings"])
            for key in ("iterations_completed", "final_metric_value", "stop_condition", "elapsed_iterations",
                        "final_rms_change"):
                if isinstance(diagnostics.get(key), list):
                    diagnostics[key] = tuple(diagnostics[key])
            application = (_config(entry["application_config"], WarpConfig) if semantics == "sequential"
                           else applications[name])
            restored.append(RegistrationResult(transform, RegistrationDiagnostics(**diagnostics), application))
        results[name] = restored
        if semantics == "recipe" and restored:
            chains[name] = TransformChain(tuple(r.transform for r in restored))
    record = {}
    if results:
        record = dict(semantics=semantics, recipe=header.get("registration_recipe"), application=applications)
    return results, chains, record


# --- Candidates stage ------------------------------------------------------------

def _detectors():
    """Saved detection method name -> config type, from SPOT_FINDING_METHODS (method values equal spec names)."""
    from starfinder._registry import config_type_for, names
    from starfinder.spot_finding import SPOT_FINDING_METHODS
    return {name: config_type_for(SPOT_FINDING_METHODS, name, "spot-finding method")
            for name in names(SPOT_FINDING_METHODS)}


def write_candidates(directory, header, spot_result, intensity_result, table_format):
    """Write candidates.<format> and candidates.json; return written file names.

    Besides the base config, candidates.json records the detection plan:
    detection_rounds (null, or the plan's rounds), detection_plan (the
    channel overrides as {channel, config} entries), execution (the
    execution entry) and weights (the provenance artifacts entries of the
    loaded weights; empty without weights). The readout_mode key comes from
    the caller's header (FOV writes the dataset's). signals.extraction_config
    keeps exactly the fields neighborhood_radius_zyx, sampling and boundary;
    the background settings are the top-level background_config (the
    LocalBackgroundConfig fields, or null when off), and, when measured, the
    top-level image_background and image_noise ({round: {channel: value}}) and
    the table's background columns (candidates_frame) hold the measurements.
    """
    from starfinder.spot_finding import _model_artifacts
    frame = candidates_frame(spot_result, intensity_result)
    table, dtypes = _write_table(frame, directory, "candidates", table_format)
    plan, diagnostics = spot_result.plan, spot_result.diagnostics
    header = dict(header, stage="candidates", format_version=FORMAT_VERSION, table=table, dtypes=dtypes,
        spot_namespace=spot_result.spot_namespace, spot_columns=list(spot_result.spots.columns),
        metadata=spot_result.metadata, detection_config=spot_result.config,
        detection_rounds=None if plan.rounds is None else list(plan.rounds),
        detection_plan=[{"channel": o.channel, "config": o.config} for o in plan.channel_overrides],
        execution=diagnostics.get("execution"),
        weights=_model_artifacts(diagnostics["model"]) if "model" in diagnostics else [],
        detection_diagnostics=_json_diagnostics(diagnostics),
        signals=None if intensity_result is None else dict(
            round_labels=intensity_result.round_labels, channel_labels=intensity_result.channel_labels,
            metadata=intensity_result.metadata, extraction_config=_extraction_fields(intensity_result.config),
            diagnostics=_json_diagnostics(intensity_result.diagnostics)))
    if intensity_result is not None:
        # New keys beside the 141c093 ones, so an older reader still loads the stage.
        measured = intensity_result.image_background is not None
        rounds, channels = intensity_result.round_labels, intensity_result.channel_labels
        header.update(background_config=intensity_result.config.background,
                      image_background=_per_round_channel(intensity_result.image_background, rounds, channels)
                      if measured else None,
                      image_noise=_per_round_channel(intensity_result.image_noise, rounds, channels)
                      if measured else None)
    write_json(header, Path(directory) / "candidates.json")
    return [table, "candidates.json"]


def _extraction_fields(config):
    """The NeighborhoodSumConfig fields a 141c093 reader rebuilds (background is a separate header key)."""
    return {name: getattr(config, name) for name in ("neighborhood_radius_zyx", "sampling", "boundary")}


def _extraction_config(header):
    """The NeighborhoodSumConfig of a candidates header; no background_config (before §2.8) is background=None."""
    from starfinder.barcode import LocalBackgroundConfig, NeighborhoodSumConfig
    background = header.get("background_config")
    fields_ = {k: _tuples(v) for k, v in header["signals"]["extraction_config"].items() if k != "method"}
    return NeighborhoodSumConfig(**fields_, background=None if background is None else LocalBackgroundConfig(
        **{k: _tuples(v) for k, v in background.items()}))


def _detection_plan(header, config, detectors):
    """The SpotFindingPlan of a candidates header; a header without the plan keys (before §2.7) gives the bare plan."""
    from starfinder.spot_finding import ChannelOverride, SpotFindingPlan
    overrides = tuple(ChannelOverride(entry["channel"], _config(entry["config"], detectors))
                      for entry in header.get("detection_plan") or ())
    rounds = header.get("detection_rounds")
    return SpotFindingPlan(config, overrides, None if rounds is None else tuple(rounds))


def _read_candidates(directory, header):
    from starfinder.barcode import IntensityExtractionResult, NeighborhoodSumConfig
    from starfinder.spot_finding import SpotFindingResult
    frame = _read_table(Path(directory) / header["table"], header["dtypes"])
    namespace = header["spot_namespace"]
    if not frame.spot_namespace.eq(namespace).all():
        raise ValueError("candidate namespace differs from the checkpoint header")
    signals = header["signals"]
    if signals is None:
        spots, values, valid, background = frame, None, None, None
    else:
        spots, values, valid = parse_candidates(frame, signals["round_labels"], signals["channel_labels"])
        background = parse_background(frame, signals["round_labels"], signals["channel_labels"])
    spots = spots[header["spot_columns"]]
    detectors = _detectors()
    config = _config(header["detection_config"], detectors)
    spot_result = SpotFindingResult(spots, _metadata(header["metadata"]), namespace, config,
        _tuples(header["detection_diagnostics"]), _detection_plan(header, config, detectors))
    if signals is None:
        return {"spot_result": spot_result, "intensity_result": None}
    rounds, channels = tuple(signals["round_labels"]), tuple(signals["channel_labels"])
    measured = {}
    if background is not None:
        bg, noise, bgvox, boxvox = background
        measured = dict(box_voxels=boxvox, background=bg, noise=noise, background_voxels=bgvox,
                        image_background=_from_round_channel(header["image_background"], rounds, channels),
                        image_noise=_from_round_channel(header["image_noise"], rounds, channels))
    intensity = IntensityExtractionResult(values, tuple(spots.spot_id), namespace,
        channels, rounds, _metadata(signals["metadata"]),
        _extraction_config(header), valid, _tuples(signals["diagnostics"]), **measured)
    return {"spot_result": spot_result, "intensity_result": intensity}


# --- Pre-QC stage ----------------------------------------------------------------

def write_pre_qc(directory, header, decoding_result, table_format, *, scoring_result=None,
                 deduplication_result=None):
    """Write the read table before the QC filter and pre_qc.json; probabilities are not saved.

    The table is the decoding table, or the scored table (the decoding columns,
    then the score columns) with scoring_result, then the deduplication columns
    with deduplication_result (whose table is the latest one). pre_qc.json
    records the result's readout_mode; a direct decoding_config carries only its
    method. The keys scoring_config and deduplication_config are the stages'
    configs (null when the stage did not run), stages_applied lists the stages
    that made the table, and layout is the caller's (FOV writes the codebook's
    segment layout, null in readout mode direct).
    """
    reads = next(r for r in (deduplication_result, scoring_result, decoding_result) if r is not None)
    table, dtypes = _write_table(reads.table.reset_index(drop=True), directory, "pre_qc", table_format)
    header = dict(header, stage="pre_qc", format_version=FORMAT_VERSION, table=table, dtypes=dtypes,
        spot_namespace=decoding_result.spot_namespace, channel_labels_decoded=decoding_result.channel_labels,
        round_labels=decoding_result.round_labels, decoding_config=decoding_result.config,
        decoding_diagnostics=_json_diagnostics(decoding_result.diagnostics),
        readout_mode=decoding_result.readout_mode,
        scoring_config=None if scoring_result is None else scoring_result.config,
        deduplication_config=None if deduplication_result is None else deduplication_result.config,
        layout=header.get("layout"),
        stages_applied=["decoding"] + ([] if scoring_result is None else ["scoring"])
        + ([] if deduplication_result is None else ["deduplication"]))
    write_json(header, Path(directory) / "pre_qc.json")
    return [table, "pre_qc.json"]


def _read_pre_qc(directory, header):
    from starfinder.barcode import (DECODING_METHODS, BarcodeDecodingResult, DeduplicationConfig,
                                    ReadDeduplicationResult, ReadScoreConfig, ReadScoringResult)
    from starfinder.barcode.deduplication import DEDUPLICATION_COLUMNS, deduplication_counts
    from starfinder.barcode.scoring import SCORE_COLUMNS
    table = _read_table(Path(directory) / header["table"], header["dtypes"])
    config = _config(header["decoding_config"], {spec.name: t for t, spec in DECODING_METHODS.items()})
    labels = tuple(header["channel_labels_decoded"]), tuple(header["round_labels"])
    scoring = deduplication = None
    if header.get("deduplication_config") is not None:
        # The deduplication columns come last; the counts are recomputed from them.
        deduplication = ReadDeduplicationResult(
            table, header["spot_namespace"], *labels, _config(header["deduplication_config"], DeduplicationConfig),
            deduplication_counts(table), readout_mode(header))
        table = table.drop(columns=list(DEDUPLICATION_COLUMNS))
    if header.get("scoring_config") is not None:
        # The decoding table is the scored table without the score columns, which come last.
        reasons = table.qc_reason
        counts = {"total": len(table), "scored": int(reasons.eq("").sum()),
                  "no_assignment": int(reasons.eq("no_assignment").sum()),
                  "background_unavailable": int(reasons.eq("background_unavailable").sum())}
        scoring = ReadScoringResult(table, header["spot_namespace"], *labels,
                                    _config(header["scoring_config"], ReadScoreConfig), counts, readout_mode(header))
        table = table.drop(columns=list(SCORE_COLUMNS))
    return {"decoding_result": BarcodeDecodingResult(table, header["spot_namespace"], *labels, config,
        _tuples(header["decoding_diagnostics"]), readout_mode(header)), "scoring_result": scoring,
        "deduplication_result": deduplication}


# --- Public reader ---------------------------------------------------------------

def read_checkpoint(path: Path | str, stage: str) -> dict:
    """Read one stage of a per-FOV checkpoint directory into existing typed results.

    Parameters
    ----------
    path : pathlib.Path or str
        Per-FOV checkpoint directory, by default ``<output_root>/checkpoints/<fov_id>``.
    stage : str
        ``registered``, ``candidates`` or ``pre_qc``.

    Returns
    -------
    dict
        Keys are the FOV attributes the stage restores. ``registered``:
        ``images`` and ``metadata`` (per round), ``snapshots`` (per round,
        the stored snapshots by name; empty for checkpoints without them), ``registration_results``,
        ``registration_chains`` (a TransformChain per registered round),
        ``registration_record`` (``semantics``: ``recipe``, or ``sequential``
        for a version-1 checkpoint, which has no chains and keeps a
        per-result application_config; the recipe summary; the application
        WarpConfig per round; empty without registration),
        ``registration_attempts`` and ``preprocessing_record`` (the recipe and
        per-round step records; empty for checkpoints written without them). ``candidates``: ``spot_result``
        (SpotFindingResult) and ``intensity_result`` (IntensityExtractionResult,
        or None when signals were not extracted); the spot result's plan is
        rebuilt from ``detection_plan`` and ``detection_rounds`` (absent in
        earlier checkpoints: no overrides and rounds None); its background
        measurements and box_voxels come from the background columns, and
        are None (with ``background=None`` in its config) for a checkpoint
        written without them. ``pre_qc``: ``decoding_result``
        (BarcodeDecodingResult without array or table diagnostics; its
        readout_mode is the header's, ``multiplexed`` when absent) and
        ``scoring_result`` (ReadScoringResult of the scored table, or None
        when the checkpoint has no ``scoring_config``) and
        ``deduplication_result`` (ReadDeduplicationResult of the deduplicated
        table, without the pair diagnostics, or None when the checkpoint has no
        ``deduplication_config``).

    Raises
    ------
    FileNotFoundError
        The stage was not written in ``path``.
    ValueError
        Unknown stage, format version or inconsistent columns.
    ImportError
        A Parquet table is read without pyarrow.
    """
    from starfinder.io.tiff import load_volume_zyxc
    directory = Path(path)
    header = read_header(directory, stage)
    if stage == "candidates":
        return _read_candidates(directory, header)
    if stage == "pre_qc":
        return _read_pre_qc(directory, header)
    images, metadata, snapshots = {}, {}, {}
    for name in header["image_rounds"]:
        loaded = load_volume_zyxc(_registered_image(directory, name),
                                  channel_labels=tuple(header["channel_labels"]))
        images[name], metadata[name] = loaded.image, loaded.metadata
        for snapshot in header.get("snapshots", []):
            snapshots.setdefault(name, {})[snapshot] = load_volume_zyxc(
                registered_image_path(directory, name, snapshot), channel_labels=tuple(header["channel_labels"])).image
    results, chains, record = _registration_results(directory, header)
    return {"images": images, "snapshots": snapshots, "metadata": metadata,
            "registration_results": results, "registration_chains": chains, "registration_record": record,
            "registration_attempts": {name: [_tuples(a) for a in attempts]
                                      for name, attempts in header["registration_attempts"].items()},
            "preprocessing_record": header.get("preprocessing") or {}}
