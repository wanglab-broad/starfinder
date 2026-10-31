"""The block translators of the Python shape: each block to the API configs its tasks run (the resolved ``api``).

Each block is translated by the kept translators of ``starfinder.dataset.workflow`` (design/specs/orchestration.md,
"The CLI"), from the block and the common part only. The plan writes every present block's result into the
resolved configuration's ``api``; a task translates its kind's blocks again and checks the result against ``api``
(``check_api``), so a package that translates differently from the one that planned is refused rather than run.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from starfinder.workflow._shape import common_part, plain

#: The processing sub-blocks a present key enables (their run flags went, D4).
_ENABLED = ("spot_finding", "reads_extraction", "reads_filtration")
#: The legacy rule the processing block is translated as (rules.<rule>.parameters).
_RULE = "rsf_single_fov"


class TranslationMismatch(ValueError):
    """A task's translation of its block differs from the resolved configuration's ``api``."""


def rotation(configuration):
    """The rotation of FOV.run and the morphology: ``rotate_angle``, with 0 meaning no rotation (OC3)."""
    angle = configuration.get("rotate_angle")
    return None if not angle else angle


def _base(configuration):
    """The dataset keys and the roots, as the kept translators read them, with the Python backend."""
    base = common_part(configuration)
    base.update({key: configuration[key] for key in ("root_input_path", "root_output_path") if key in configuration})
    base["backend"] = "python"
    base["rotate_angle"] = rotation(configuration)
    return base


def rule_configuration(configuration):
    """The processing block as the kept translator reads it: rules.rsf_single_fov.parameters with the run flags.

    The present sub-blocks spot_finding, reads_extraction and reads_filtration are enabled; the codebook is loaded
    whenever the block decodes (load_codebook always runs); deduplication is on when present; ``scoring: false``
    is ``{run: false}``; reference_image_view.projection is the rule's maximum_projection; mode, tiled and
    checkpoints are the plan's (the checkpoints are passed to FOV.run by the fov command).
    """
    block = configuration["processing"]
    params = copy.deepcopy({k: v for k, v in block.items() if k not in ("mode", "tiled", "checkpoints")})
    for name in _ENABLED:
        if name in params:
            params[name] = {"run": True, **(params[name] or {})}
    if "deduplication" in params:
        params["deduplication"] = {"run": True, **(params["deduplication"] or {})}
    if params.get("scoring") is False:
        params["scoring"] = {"run": False}
    view = dict(params.get("reference_image_view") or {})
    projection = view.pop("projection", None)
    if view:
        params["reference_image_view"] = view
    else:
        params.pop("reference_image_view", None)
    params["load_codebook"] = {"run": True, **(params.get("load_codebook") or {})}
    rule = dict(_base(configuration), maximum_projection=projection == "max", workflow_mode="free",
                rules={_RULE: {"run": True, "parameters": params}})
    return rule


@dataclass(frozen=True)
class Processing:
    """The processing block translated: the WorkflowConfig of the kept translator and the checkpoint stages."""
    workflow: object
    stages: tuple


def processing(configuration) -> Processing:
    """The processing block's translation (from_workflow_config of rsf_single_fov on the rule view)."""
    from starfinder.dataset.workflow import from_workflow_config
    adapted = from_workflow_config(rule_configuration(configuration), _RULE)
    return Processing(adapted, tuple(configuration["processing"].get("checkpoints") or ()))


def _processing_api(configuration):
    translated = processing(configuration)
    adapted = translated.workflow
    return {"pipeline": plain(adapted.pipeline), "execution": plain(adapted.execution),
            "checkpoints": {"stages": list(translated.stages)},
            "reference_image_view": {"projection": plain(adapted.reference_projection),
                                     "image": adapted.reference_image, "channel": adapted.reference_channel},
            "codebook": {"encoding": plain(adapted.encoding), "layout": plain(adapted.layout)}}


def _morphology_api(configuration):
    from starfinder.dataset.config import MorphologyConfig
    rounds = tuple(configuration["morphology"]["rounds"])
    return plain(MorphologyConfig(rotation_degrees=rotation(configuration), rounds=rounds))


def _segmentation_api(configuration):
    from starfinder.dataset.workflow import _segmentation_block
    plan_, device = _segmentation_block(dict(_base(configuration), segmentation=configuration["segmentation"]))
    return {"plan": plain(plan_), "device": device}


def _assignment_api(configuration):
    from starfinder.dataset.workflow import _assignment_block
    values = {k: v for k, v in configuration["assignment"].items() if k != "previews"}
    config, call = _assignment_block(dict(_base(configuration), assignment=values))
    return {"config": plain(config), "call": plain(call), "previews": configuration["assignment"]["previews"]}


def stitching(configuration):
    """The stitching block translated (_stitching_block for the task commands)."""
    from starfinder.dataset.workflow import _stitching_block
    return _stitching_block(dict(_base(configuration), stitching=configuration["stitching"]), tasks=True)


def _stitching_api(configuration):
    block = stitching(configuration)
    return plain({"method": block.method, "config": block.config, "input": block.input,
                  "positions": block.positions, "review": block.review, "fallback": block.fallback,
                  "bigstitcher_save_dir": block.bigstitcher_save_dir, "dataset_xml": block.dataset_xml,
                  "view_map": block.view_map})


def export(configuration):
    """(ExportConfig, assembly block) of the assembly and export blocks (_export_config), without the previews."""
    from starfinder.dataset.workflow import _export_config
    values = {k: v for k, v in (configuration.get("export") or {}).items() if k != "previews"}
    config = dict(_base(configuration), stitching=configuration["stitching"], assembly=configuration["assembly"],
                  export=values)
    if configuration.get("assignment") is not None:
        config["assignment"] = {k: v for k, v in configuration["assignment"].items() if k != "previews"}
    return _export_config(config)


def route(export_config, assembly):
    """The export route: molecules_only without an assignment name, else resolve_route's (export/_segment.py)."""
    from starfinder.export._segment import resolve_route
    return "molecules_only" if assembly.assignment is None else resolve_route(export_config)[0]


def _export_api(configuration):
    from starfinder.dataset.workflow import _check_export_route
    config, assembly = export(configuration)
    values = {k: v for k, v in (configuration.get("export") or {}).items() if k != "previews"}
    check = dict(_base(configuration), stitching=configuration["stitching"], assembly=configuration["assembly"],
                 export=values)
    if configuration.get("assignment") is not None:
        check["assignment"] = {k: v for k, v in configuration["assignment"].items() if k != "previews"}
    _check_export_route(check)
    return {"config": plain(config), "assignment": assembly.assignment, "route": route(config, assembly),
            "previews": configuration["export"]["previews"]}


def _report_api(configuration):
    from starfinder.dataset.workflow import _report_block
    return {"config": plain(_report_block(dict(_base(configuration), report=configuration["report"])))}


TRANSLATORS = {"processing": _processing_api, "morphology": _morphology_api, "segmentation": _segmentation_api,
               "assignment": _assignment_api, "stitching": _stitching_api, "export": _export_api,
               "report": _report_api}


def api(configuration) -> dict:
    """The ``api`` of a resolved configuration: each present block's translation, keyed by block.

    ``export`` holds the assembly and export blocks' translation (the assembly block has none of its own). The
    values that a kind's commands need from another block are added by ``cross_block``.
    """
    out = {}
    for block, translate in TRANSLATORS.items():
        if configuration.get(block) is not None or (block == "export" and configuration.get("assembly") is not None):
            out[block] = translate(configuration)
    out.update(cross_block(configuration, out))
    return out


#: The (block, key) of api that cross_block adds.
_CROSS_BLOCK = {("export", "codebook"), ("report", "assignment")}


def cross_block(configuration, out):
    """The values a kind's commands read from another block, resolved by the plan into their own block's ``api``.

    The export of the molecule tables (routes molecules_only and after_fusion) loads the codebook for its genes with
    the processing block's encoding and layout (``export.codebook``); the report reads the assembly block's
    assignment name (``report.assignment``). The task's fingerprint follows them through its ``needs``.
    """
    extra = {}
    if "export" in out:
        codebook = out["processing"]["codebook"] if "processing" in out else None
        extra["export"] = dict(out["export"], codebook=codebook if out["export"]["route"] != "per_fov" else None)
    if "report" in out:
        assignment = out["export"]["assignment"] if "export" in out else None
        extra["report"] = dict(out["report"], assignment=assignment)
    return extra


def check_api(view, api_view, blocks, kind):
    """Translate ``blocks`` again from the configuration ``view`` and compare with ``api_view`` (exit 2 on a
    difference): the values of other blocks (``cross_block``) are compared as the plan wrote them."""
    configuration = view.plain()
    for block in blocks:
        present = configuration.get("assembly" if block == "export" else block) is not None
        if block == "assembly" or not present:
            continue
        again = TRANSLATORS[block](configuration)
        recorded = {k: v for k, v in api_view[block].items() if (block, k) not in _CROSS_BLOCK}
        if again != recorded:
            difference = sorted(k for k in set(again) | set(recorded) if again.get(k) != recorded.get(k))
            raise TranslationMismatch(
                f"the {kind} task translates the {block} block differently from the plan: api.{block} differs in "
                f"{', '.join(difference)}; plan the run again with this package")
