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

from starfinder.preprocessing.histograms import HistogramSummary, _labels, _steps_record
from starfinder.preprocessing.steps import (PREPROCESSING_METHODS, PreprocessingRecipe, _supplied, step_config_type,
    step_spec)

_SCHEMA = "starfinder.preprocessing.supplied/1"
_ENVELOPE = {"schema", "dtype", "channel_labels", "fovs_used", "fovs_excluded", "steps"}
_SECTION = {"summarized_after", "params", "fitted"}


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
    """Fit one step's supplied section from merged histograms, with the step's SuppliedSpec.fit hook.

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
    spec = step_spec(config)
    if spec.supplied is None:
        raise ValueError(f"step {spec.name!r} has no supplied statistics")
    # Only a step fitted on the reference round alone (histogram matching) takes reference_round.
    if spec.supplied.per_round and reference_round is not None:
        raise ValueError("reference_round applies only to histogram matching")
    return {"summarized_after": after, **spec.supplied.fit(config, merged, reference_round)}


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
        supplied = PREPROCESSING_METHODS[step_config_type(name)].supplied
        if supplied is None:
            raise ValueError(f"step {name!r} has no supplied statistics")
        if not isinstance(section, dict) or set(section) != _SECTION:
            raise ValueError(f"section {name!r} must have exactly the keys {sorted(_SECTION)}")
        _steps_record(section["summarized_after"])
        if not isinstance(section["fitted"], dict) or not section["fitted"]:
            raise ValueError(f"section {name!r} must give fitted values by round")
        supplied.validate(section, dtype, len(labels))
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
        spec = step_spec(config)
        name = spec.name
        section = document["steps"].get(name)
        if section is None:
            raise ValueError(f"supplied statistics have no section for step {name!r}")
        expected = _preceding(recipe, index)
        if section["summarized_after"] != expected:
            raise ValueError(f"section {name!r} was summarized after {section['summarized_after']}, "
                             f"but the recipe runs {expected} before it")
        spec.supplied.check_params(config, section["params"], name)
        if spec.supplied.per_round:
            missing = [r for r in rounds or () if r not in section["fitted"]]
            if missing:
                raise ValueError(f"section {name!r} has no values for rounds {missing}")
    return document
