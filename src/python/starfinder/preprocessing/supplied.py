"""Supplied-statistics files (starfinder.preprocessing.supplied/1) for fit="supplied" steps.

A generic envelope (schema, dtype, channel labels, FOVs used and excluded) is
validated once; each fitted step has one section keyed by its step name,
recording summarized_after, params and fitted by round, and validates it
itself. See the preprocessing algorithms page.
"""
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from starfinder.preprocessing.background import ScalarBackgroundConfig, _supplied_background
from starfinder.preprocessing.histograms import HistogramSummary, _labels, _steps_record, histogram_percentile
from starfinder.preprocessing.normalization import (HistogramMatchingConfig, PercentileNormalizationConfig,
    _supplied_range)
from starfinder.preprocessing.steps import PreprocessingRecipe, _supplied, step_config_type, step_spec

_SCHEMA = "starfinder.preprocessing.supplied/1"
_ENVELOPE = {"schema", "dtype", "channel_labels", "fovs_used", "fovs_excluded", "steps"}
_SECTION = {"summarized_after", "params", "fitted"}


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _preceding(recipe, index):
    return _steps_record({"step": step_spec(entry.config).name, "config": asdict(entry.config)}
                         for entry in recipe.steps[:index])


def _supplied_index(recipe, name):
    for index, entry in enumerate(recipe.steps):
        if _supplied(entry.config) and step_spec(entry.config).name == name:
            return index
    raise ValueError(f'the recipe has no step {name!r} with fit="supplied"')


def summary_stage(recipe: PreprocessingRecipe, step: str) -> tuple[PreprocessingRecipe, tuple[dict, ...]]:
    """Recipe to run for the summary pass of a supplied step, and its summarized_after record.

    Statistics for the step named step (with fit="supplied") are summarized
    at its input: after every preceding step of the recipe, each in its own
    fit mode. The returned recipe holds exactly those preceding steps (with
    the same supplied_statistics file, needed when an earlier step is itself
    supplied, no post_registration steps and no extraction or registration
    source, since the summary pass neither extracts nor registers). The record lists them as
    {"step": name, "config": {...}} for summarize_histograms.

    Raises
    ------
    ValueError
        The recipe has no step of that name with fit="supplied".
    """
    if not isinstance(recipe, PreprocessingRecipe):
        raise TypeError("summary_stage requires a PreprocessingRecipe")
    index = _supplied_index(recipe, step)
    prefix = recipe.steps[:index]
    path = recipe.supplied_statistics if any(_supplied(entry.config) for entry in prefix) else None
    return (replace(recipe, steps=prefix, post_registration=(), extraction_source=None, registration_source=None,
                    supplied_statistics=path), tuple(_preceding(recipe, index)))


def supplied_section(config, merged: HistogramSummary, *, reference_round: str | None = None) -> dict:
    """Fit one step's supplied section from merged histograms.

    PercentileNormalizationConfig: params {"p_low", "p_high"} and, per
    round, {"low": [...], "high": [...]} from the merged counts by the
    inverted-CDF definition. ScalarBackgroundConfig: params {"percentile"}
    and, per round, {"background": [...]} likewise. HistogramMatchingConfig: params
    {"reference_round", "reference_channel"} and, for reference_round only,
    the merged count vector of reference_channel as {"values", "counts"}
    (nonzero bins only). summarized_after is copied from merged.

    Raises
    ------
    ValueError
        A config type without supplied statistics, a histogram reference
        round or channel missing from merged, or reference_round given for
        another step.
    """
    if not isinstance(merged, HistogramSummary):
        raise TypeError("supplied_section requires a merged HistogramSummary")
    config.__post_init__()
    after = list(merged.summarized_after)
    if type(config) is PercentileNormalizationConfig:
        if reference_round is not None:
            raise ValueError("reference_round applies only to histogram matching")
        fitted = {name: {"low": [histogram_percentile(counts, config.p_low) for counts in merged.counts[r]],
                         "high": [histogram_percentile(counts, config.p_high) for counts in merged.counts[r]]}
                  for r, name in enumerate(merged.round_names)}
        return {"summarized_after": after, "params": {"p_low": float(config.p_low), "p_high": float(config.p_high)},
                "fitted": fitted}
    if type(config) is ScalarBackgroundConfig:
        if reference_round is not None:
            raise ValueError("reference_round applies only to histogram matching")
        fitted = {name: {"background": [histogram_percentile(counts, config.percentile) for counts in merged.counts[r]]}
                  for r, name in enumerate(merged.round_names)}
        return {"summarized_after": after, "params": {"percentile": float(config.percentile)}, "fitted": fitted}
    if type(config) is HistogramMatchingConfig:
        if reference_round not in merged.round_names:
            raise ValueError(f"reference_round {reference_round!r} is not a summarized round {list(merged.round_names)}")
        if config.reference_channel >= len(merged.channel_labels):
            raise ValueError(f"reference_channel {config.reference_channel} is outside the summarized channels")
        counts = merged.counts[merged.round_names.index(reference_round), config.reference_channel]
        values = np.flatnonzero(counts)
        return {"summarized_after": after,
                "params": {"reference_round": reference_round, "reference_channel": config.reference_channel},
                "fitted": {reference_round: {"values": values.tolist(), "counts": counts[values].tolist()}}}
    raise ValueError(f"step {step_spec(config).name!r} has no supplied statistics")


def _percentile_section(section, dtype, n_channels):
    params = section["params"]
    if not isinstance(params, Mapping) or set(params) != {"p_low", "p_high"}:
        raise ValueError('percentile_normalization params must be {"p_low", "p_high"}')
    PercentileNormalizationConfig(params["p_low"], params["p_high"])
    for name, entry in section["fitted"].items():
        try:
            _supplied_range(entry, n_channels)
        except ValueError as error:
            raise ValueError(f"percentile_normalization round {name!r}: {error}") from None


def _scalar_section(section, dtype, n_channels):
    params = section["params"]
    if not isinstance(params, Mapping) or set(params) != {"percentile"}:
        raise ValueError('scalar_background params must be {"percentile"}')
    ScalarBackgroundConfig(params["percentile"])
    for name, entry in section["fitted"].items():
        try:
            _supplied_background(entry, n_channels)
        except ValueError as error:
            raise ValueError(f"scalar_background round {name!r}: {error}") from None


def _histogram_section(section, dtype, n_channels):
    params = section["params"]
    if not isinstance(params, Mapping) or set(params) != {"reference_round", "reference_channel"}:
        raise ValueError('histogram_matching params must be {"reference_round", "reference_channel"}')
    round_name, channel = params["reference_round"], params["reference_channel"]
    if not _integer(channel) or not 0 <= channel < n_channels:
        raise ValueError(f"histogram_matching reference_channel {channel!r} is not one of the {n_channels} channels")
    if dtype.kind != "u":
        raise ValueError(f"a supplied histogram reference requires unsigned integer data, not {dtype}")
    if set(section["fitted"]) != {round_name}:
        raise ValueError(f"histogram_matching fitted values must be given for reference round {round_name!r} only")
    entry = section["fitted"][round_name]
    if not isinstance(entry, Mapping) or set(entry) != {"values", "counts"}:
        raise ValueError('histogram_matching fitted values must be {"values", "counts"}')
    values, counts = entry["values"], entry["counts"]
    if not isinstance(values, list) or not isinstance(counts, list) or not values or len(values) != len(counts) \
            or not all(map(_integer, values + counts)):
        raise ValueError("histogram_matching values and counts must be nonempty integer lists of equal length")
    if min(counts) <= 0 or values[0] < 0 or values[-1] > np.iinfo(dtype).max or any(np.diff(values) <= 0):
        raise ValueError("histogram_matching values must increase within the dtype range, with positive counts")


_SECTION_VALIDATORS = {PercentileNormalizationConfig: _percentile_section, ScalarBackgroundConfig: _scalar_section,
                       HistogramMatchingConfig: _histogram_section}


def _validate(document):
    """Validate the envelope and every section; return the JSON-normalized document."""
    try:
        document = json.loads(json.dumps(document, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"supplied statistics must be JSON-serializable: {error}") from None
    if not isinstance(document, dict) or set(document) != _ENVELOPE:
        raise ValueError(f"supplied statistics must have exactly the keys {sorted(_ENVELOPE)}")
    if document["schema"] != _SCHEMA:
        raise ValueError(f"unsupported supplied-statistics schema {document['schema']!r}; expected {_SCHEMA!r}")
    try:
        dtype = np.dtype(document["dtype"])
    except TypeError:
        raise ValueError(f"unknown dtype {document['dtype']!r}") from None
    if dtype.kind not in "uif" or dtype.name != document["dtype"]:
        raise ValueError(f"unsupported dtype {document['dtype']!r}")
    for key in ("channel_labels", "fovs_used", "fovs_excluded"):
        if not isinstance(document[key], list):
            raise ValueError(f"{key} must be a list of names")
    labels = _labels(document["channel_labels"], "channel_labels")
    used = _labels(document["fovs_used"], "fovs_used")
    if document["fovs_excluded"]:
        _labels(document["fovs_excluded"], "fovs_excluded")
    if set(used) & set(document["fovs_excluded"]):
        raise ValueError("a FOV cannot be both used and excluded")
    if not isinstance(document["steps"], dict):
        raise ValueError("steps must map step names to sections")
    for name, section in document["steps"].items():
        validator = _SECTION_VALIDATORS.get(step_config_type(name))
        if validator is None:
            raise ValueError(f"step {name!r} has no supplied statistics")
        if not isinstance(section, dict) or set(section) != _SECTION:
            raise ValueError(f"section {name!r} must have exactly the keys {sorted(_SECTION)}")
        _steps_record(section["summarized_after"])
        if not isinstance(section["fitted"], dict) or not section["fitted"]:
            raise ValueError(f"section {name!r} must give fitted values by round")
        validator(section, dtype, len(labels))
    return document


def supplied_statistics(merged: HistogramSummary, sections: Mapping[str, Mapping]) -> dict[str, Any]:
    """Supplied-statistics document: the envelope from merged and sections keyed by step name.

    The envelope takes dtype, channel labels and the FOVs used and excluded
    from merged. Sections come from supplied_section; for a later summary
    pass, add its section to the document's "steps" and write it again.

    Raises
    ------
    ValueError
        An unknown step name or an invalid section.
    """
    if not isinstance(merged, HistogramSummary):
        raise TypeError("supplied_statistics requires a merged HistogramSummary")
    return _validate({"schema": _SCHEMA, "dtype": merged.dtype, "channel_labels": list(merged.channel_labels),
                      "fovs_used": list(merged.fovs_used), "fovs_excluded": list(merged.fovs_excluded),
                      "steps": dict(sections)})


def write_supplied_statistics(statistics: Mapping[str, Any], path: Path | str) -> Path:
    """Validate a supplied-statistics document and write it as JSON in one full-file write."""
    document = _validate(dict(statistics))
    path = Path(path)
    path.write_text(json.dumps(document, indent=1) + "\n")
    return path


def read_supplied_statistics(path: Path | str, recipe: PreprocessingRecipe | None = None, *, dtype: str | None = None,
                             channel_labels: Sequence[str] | None = None,
                             rounds: Sequence[str] | None = None) -> dict[str, Any]:
    """Read and validate a supplied-statistics file.

    The envelope and every section are always validated. When given, dtype
    and channel_labels must equal the file's. With a recipe, every step with
    fit="supplied" must have a section whose summarized_after equals the
    recipe's preceding steps and whose params equal the step's config
    (p_low and p_high; percentile; reference_channel). With rounds, a
    percentile normalization or scalar background section must give values
    for each of them. Missing rounds
    and channel counts are also checked when a step runs.

    Raises
    ------
    ValueError
        A wrong schema, dtype or channel labels; a missing section, round or
        channel; a summarized_after or params that differ from the recipe.
    """
    try:
        document = json.loads(Path(path).read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"supplied statistics {path} are not valid JSON: {error}") from None
    document = _validate(document)
    if dtype is not None and document["dtype"] != np.dtype(dtype).name:
        raise ValueError(f"supplied statistics are for dtype {document['dtype']}, not {np.dtype(dtype).name}")
    if channel_labels is not None and tuple(document["channel_labels"]) != tuple(channel_labels):
        raise ValueError(f"supplied channel labels {document['channel_labels']} differ from {list(channel_labels)}")
    if recipe is None:
        return document
    if not isinstance(recipe, PreprocessingRecipe):
        raise TypeError("recipe must be a PreprocessingRecipe")
    for index, entry in enumerate(recipe.steps):
        config = entry.config
        if not _supplied(config):
            continue
        name = step_spec(config).name
        section = document["steps"].get(name)
        if section is None:
            raise ValueError(f"supplied statistics have no section for step {name!r}")
        expected = _preceding(recipe, index)
        if section["summarized_after"] != expected:
            raise ValueError(f"section {name!r} was summarized after {section['summarized_after']}, "
                             f"but the recipe runs {expected} before it")
        params = section["params"]
        if type(config) is PercentileNormalizationConfig:
            if (params["p_low"], params["p_high"]) != (config.p_low, config.p_high):
                raise ValueError(f"section {name!r} was fitted with p_low={params['p_low']}, p_high={params['p_high']}, "
                                 f"not the recipe's {config.p_low}, {config.p_high}")
        elif type(config) is ScalarBackgroundConfig:
            if params["percentile"] != config.percentile:
                raise ValueError(f"section {name!r} was fitted with percentile={params['percentile']}, "
                                 f"not the recipe's {config.percentile}")
        elif params["reference_channel"] != config.reference_channel:
            raise ValueError(f"section {name!r} summarized reference_channel {params['reference_channel']}, "
                             f"not the recipe's {config.reference_channel}")
        if type(config) is not HistogramMatchingConfig:
            missing = [r for r in rounds or () if r not in section["fitted"]]
            if missing:
                raise ValueError(f"section {name!r} has no values for rounds {missing}")
    return document
