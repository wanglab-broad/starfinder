"""Every legacy key and value alias of W-569's inventory is refused by the Python backend, in every position.

INVENTORY transcribes the rows of the inventory saved before the legacy translation left the Python path (the keys
and value aliases the base revision accepted, with their native forms). One test checks that the parse-time
tables of starfinder.dataset.workflow and the lists of the schema's backend guard hold exactly these rows, so that no
row is missed in either; the other runs every row in every position it applies to (an active rule, a rule that does
not run, a disabled block, a registration step, a recovery alternative, a channel override) and asserts that the
parse-time check names the native form and that the Snakefile's validation (with default insertion) refuses it.
"""
import copy
import re
from pathlib import Path

import pytest
import yaml

from starfinder.dataset import from_workflow_config
from starfinder.dataset import workflow

from .test_workflow_config_validation import snakefile_errors

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow" / "schemas" / "config.schema.yaml").read_text())
MINIMAL = yaml.safe_load((ROOT / "docs" / "examples" / "workflow-minimal.yaml").read_text())

# The inventory's rows, by where the key sits: (example value, a fragment of the native form the error names).
INVENTORY = {
    # Section 1, rows 1-8: keys of a processing rule's parameters.
    "rule": {"enhance_contrast": ({"run": True}, "min_max_normalization"),
             "hist_equalize": ({"run": True, "reference_channel": 0}, "histogram_matching"),
             "morph_recon": ({"run": True, "radius": 3}, "reconstruction"),
             "tophat": ({"run": True, "radius": 3}, "white_tophat"),
             "snr_threshold": (3.0, "min_max_normalization"),
             "global_registration": ({"run": True}, "registration.steps"),
             "local_registration": ({"run": False}, "registration.steps")},
    # Rows 9-22: keys of the registration blocks, refused in the native steps and recovery alternatives.
    "registration": {"ref_img": ("merged-image", "the signal"), "mov_img": ("merged-image", "the signal"),
                     "ref_channel": (0, "the signal reference_channel"),
                     "boundary_mode": ("nearest", "registration.warp"),
                     "ref_round": ("round1", "registration.reference_round"),
                     "detection_threshold": (3.0, "detection_noise_sigma"),
                     "match_distance": (2.0, "match_distance_voxels"), "tps_smoothing": (0.5, "smoothing"),
                     "grid_spacing": (32, "grid_spacing_voxels"), "beta": (2.0, "kernel_width_voxels"),
                     "lmbda": (2.0, "regularization_weight"), "cpd_w": (0.1, "outlier_fraction"),
                     "candidate_radius": (3, "candidate_radius_voxels"), "k_neighbors": (4, "neighbors_per_anchor")},
    # Rows 23-25: local_maxima's spot_finding keys.
    "spot": {"intensity_estimation": ("noise", "threshold_mode"), "intensity_threshold": (5.0, "threshold_value"),
             "min_distance": (2, "min_distance_voxels")},
    # Section 2: the value aliases.
    "variant": {name: (name, f"{{method: demons, variant: {name}}}")
                for name in ("diffeomorphic", "symmetric", "fast_symmetric")},
    "signal": {"merged-image": ("merged-image", "mode 'max'"), "merged": ("merged", "mode 'max'"),
               "single-channel": ("single-channel", "mode 'channel'")},
    # The block fields that the native steps rename (morph_recon.radius, tophat.radius), and the top-level stain key.
    "preprocessing": {"radius": (3, "radius_yx")},
    "top": {"dapi_round": ("round1", "reference_stains")},
}
RECOVERY = {"allowed_errors": ["InsufficientLandmarksError"]}


def schema_guard():
    """(rule keys, spot keys, registration names, variant methods) of the schema's backend guard."""
    clause = next(c for c in SCHEMA["allOf"] if c.get("then", {}).get("not", {}).get("required") == ["dapi_round"])
    entries = clause["then"]["properties"]["rules"]["additionalProperties"]["properties"]["parameters"]["not"]["anyOf"]
    rule = {e["required"][0] for e in entries if set(e) == {"required"}}
    spot = set(re.findall(r"'required': \['(intensity_estimation|intensity_threshold|min_distance)'\]",
                          repr([e for e in entries if e["required"] == ["spot_finding"]])))
    names = SCHEMA["$defs"]["matlab_registration_names"]["anyOf"]
    registration = {e["required"][0] for e in names if e["required"] != ["method"]}
    variants = next(e for e in names if e["required"] == ["method"])["properties"]["method"]["enum"]
    return rule, spot, registration, tuple(variants)


def test_the_parse_time_tables_and_the_schema_guard_hold_the_inventory():
    rule, spot, registration, variants = schema_guard()
    assert set(workflow._NATIVE_FORMS) == rule == set(INVENTORY["rule"])
    assert set(workflow._NATIVE_SPOT_FIELDS) == spot == set(INVENTORY["spot"])
    assert set(workflow._NATIVE_PARAMETERS) | set(workflow._NATIVE_BLOCK_KEYS) == registration \
        == set(INVENTORY["registration"])
    assert workflow._VARIANT_METHODS == variants == tuple(INVENTORY["variant"])
    assert set(workflow._MATLAB_SIGNALS) == set(INVENTORY["signal"])
    assert not set(INVENTORY["signal"]) & set(SCHEMA["$defs"]["registration_signal"]["properties"]["mode"]["enum"])


def place(position, parameters):
    """The minimal example (backend python) with the rule parameters in the active rule or in a rule that does not
    run (gr_single_fov_subtile; deep_rsf_subtile for spot_finding, which gr_single_fov_subtile does not declare)."""
    config = copy.deepcopy(MINIMAL)
    if position == "active rule":
        config["rules"]["rsf_single_fov"]["parameters"].pop("registration")
        config["rules"]["rsf_single_fov"]["parameters"].update(parameters)
    else:
        rule = "deep_rsf_subtile" if "spot_finding" in parameters else "gr_single_fov_subtile"
        config["rules"][rule] = {"run": False, "parameters": parameters}
    return config


def cases():
    """(id, configuration, native fragment) of every inventory row in every position it applies to."""
    out = []
    rules = ("active rule", "disabled rule")
    for key, (value, native) in INVENTORY["rule"].items():
        for position in rules:
            out.append((f"rule/{key}/{position}", place(position, {key: value}), native))
    for key, (value, native) in INVENTORY["registration"].items():
        for position in rules:
            out.append((f"registration/{key}/step/{position}",
                        place(position, {"registration": {"steps": [{"method": "cpd", key: value}]}}), native))
            out.append((f"registration/{key}/recovery alternative/{position}", place(position, {"registration": {
                "steps": [{"method": "tps", "recovery": {**RECOVERY, "alternatives": [{"method": "cpd", key: value}]}}]}}),
                native))
    for key, (value, native) in INVENTORY["variant"].items():
        for position in rules:
            out.append((f"variant/{key}/step/{position}",
                        place(position, {"registration": {"steps": [{"method": value}]}}), native))
            out.append((f"variant/{key}/recovery alternative/{position}", place(position, {"registration": {
                "steps": [{"method": "tps", "recovery": {**RECOVERY, "alternatives": [{"method": value}]}}]}}), native))
    for key, (value, native) in INVENTORY["signal"].items():
        for position in rules:
            out.append((f"signal/{key}/recipe/{position}", place(position, {"registration": {
                "steps": [{"method": "translation"}], "signal": {"mode": value}}}), native))
            out.append((f"signal/{key}/step/{position}", place(position, {"registration": {
                "steps": [{"method": "translation", "signal": {"mode": value}}]}}), native))
    for key, (value, native) in INVENTORY["spot"].items():
        for run in (True, False):
            out.append((f"spot/{key}/block run={run}", place("active rule", {"spot_finding": {"run": run, key: value}}),
                        native))
            out.append((f"spot/{key}/channel override run={run}", place("active rule", {"spot_finding": {
                "run": run, "channel_overrides": {"ch02": {key: value}}}}), native))
        out.append((f"spot/{key}/disabled rule", place("disabled rule", {"spot_finding": {"run": True, key: value}}),
                    native))
    for key, (value, native) in INVENTORY["preprocessing"].items():
        for method in ("reconstruction", "white_tophat"):
            for where in ("steps", "post_registration"):
                for position in rules:
                    preprocessing = {"steps": [], where: [{"method": method, key: value}]}
                    out.append((f"preprocessing/{key}/{method} in {where}/{position}",
                                place(position, {"preprocessing": preprocessing}), native))
    for key, (value, native) in INVENTORY["top"].items():
        out.append((f"top/{key}", dict(copy.deepcopy(MINIMAL), **{key: value}), native))
    return out


CASES = cases()


@pytest.mark.parametrize("config, native", [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_every_inventory_row_raises_in_every_position_naming_its_native_form(config, native):
    for check in (workflow._check_python_dialect, from_workflow_config):
        with pytest.raises(ValueError, match=re.escape(native)):
            check(copy.deepcopy(config))
    assert snakefile_errors(copy.deepcopy(config))


def test_the_shared_ref_round_keys_stay_valid():
    """ref_round is refused in the registration steps and alternatives only: the top-level key and spot_finding's
    stay shared keys (checked against ref_round)."""
    config = place("active rule", {"spot_finding": {"run": True, "ref_round": "round1"},
                                   "registration": {"steps": [{"method": "translation"}], "reference_round": "round1"}})
    workflow._check_python_dialect(config)
    assert from_workflow_config(config).pipeline.registration.reference_round == "round1"
    assert snakefile_errors(copy.deepcopy(config)) == []
    assert snakefile_errors(copy.deepcopy(MINIMAL)) == []
