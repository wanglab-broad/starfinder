"""The sequencing logs folded into run.json and the morphology registration record (W-563, OM-05 and OM-17).

design/notes/migration.md, "Sequencing logs and the morphology record". A checkpointed ``FOV.run`` records in ``run.json``
the filter's ``counts.filtering.fractions``, the read summary ``counts.summary`` whenever reads were decoded (with or
without scoring and deduplication) and the round roles ``rounds``; ``FOV.save_registration_record`` writes
``fovs/<fov_id>/nuclei_registration.json`` with the registered other rounds and their attempts, unchanged. The final
targets of ``common.smk`` branch on ``backend``: each target is an output of a rule of that backend's rule files, the
Python processing rules request their run.json and no log, and the MATLAB rules keep theirs. Scenes: the 4 x 24 x 24
FOV of test_report_records_golden.py and the 8 x 32 x 32 other-round FOV of test_registration_other_rounds.py.
"""
from dataclasses import asdict, replace
import itertools
import json
import re
from pathlib import Path

import pytest

from starfinder._snakemake import _reference_image_outputs, _run_record_outputs
from starfinder.readout import summarize_reads

from .test_registration_other_rounds import TRANSLATION, external, other_round_fov
from .test_report_records_golden import ALL_STAGES, pipeline
from .test_report_run_record_additions import golden_dataset, jsonable, run
from .test_report_workflow import SAMPLE, annotation, common_preamble, repository_config

pytestmark = [pytest.mark.dataset, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]


# --- run.json -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("stages", [(), ALL_STAGES])
@pytest.mark.parametrize("scored", [False, True])
def test_run_json_holds_the_fractions_the_summary_and_the_round_roles(tmp_path, stages, scored):
    config = pipeline() if scored else replace(pipeline(), scoring=None, deduplication=None)
    fov, record, _ = run(golden_dataset(tmp_path), stages, config)
    assert (fov.scoring_result is not None) is scored and (fov.deduplication_result is not None) is scored
    counts = record["counts"]
    # The summary is written whenever reads were decoded, also without scoring or deduplication.
    assert counts["summary"] == jsonable(summarize_reads(fov.results))
    assert counts["summary"]["filtering"]["accepted"] == counts["filtering"]["accepted"]
    # The filter's fractions, as ReadFilteringResult.fractions computes them (end_bases: with endpoint_valid).
    assert counts["filtering"]["fractions"] == jsonable(fov.filtering_result.fractions)
    assert set(counts["filtering"]["fractions"]) == {"accepted", "endpoint_valid"}
    total = counts["filtering"]["total"]
    assert counts["filtering"]["fractions"]["accepted"] == pytest.approx(counts["filtering"]["accepted"] / total)
    assert counts["filtering"]["fractions"]["endpoint_valid"] == pytest.approx(
        counts["filtering"]["endpoint_valid"] / total)
    assert record["rounds"] == jsonable(asdict(fov.rounds)) == {
        "sequencing_rounds": ["round1", "round2"], "other_rounds": [], "reference_round": "round1"}
    # The shift rows of the former shift table: each translation result's displacement, (dz, dy, dx).
    shifts = {name: [list(r.transform.displacement_zyx) for r in results]
              for name, results in fov.registration_results.items()}
    assert {name: [r["transform"]["displacement_zyx"] for r in entry["results"]]
            for name, entry in record["transforms"].items()} == shifts == {"round2": [[0, 0, 1]]}
    assert record["registration"] == jsonable(fov.registration_attempts)
    assert record["format_version"] == 2


def test_run_json_has_no_summary_or_fractions_before_decoding(tmp_path):
    config = replace(pipeline(), decoding=None, scoring=None, deduplication=None, filtering=None)
    fov, record, _ = run(golden_dataset(tmp_path), (), config)
    assert fov.decoding_result is None
    assert set(record["counts"]) == {"spots", "intensities"}
    assert record["rounds"]["reference_round"] == "round1"


# --- nuclei_registration.json ---------------------------------------------------------------------------------

@pytest.mark.parametrize("use_external", [False, True])
def test_save_registration_record_writes_the_attempts_unchanged(tmp_path, use_external):
    fov, _, _ = other_round_fov(tmp_path)
    fov.register(TRANSLATION, rounds=["morph"], reference=external(fov) if use_external else None)
    path = fov.save_registration_record()
    assert path == tmp_path / "output" / "fovs" / "FOV" / "nuclei_registration.json"
    record = json.loads(path.read_text())
    assert list(record) == ["fov_id", "rounds", "registered_rounds", "registration"]
    assert record["fov_id"] == "FOV"
    assert record["rounds"] == {"sequencing_rounds": ["round1", "round2"], "other_rounds": ["morph"],
                                "reference_round": "round1"}
    assert record["registered_rounds"] == {"morph": jsonable(fov.registration_record["rounds"]["morph"])}
    assert record["registration"] == {"morph": jsonable(fov.registration_attempts["morph"])}
    estimation = record["registration"]["morph"][0]
    assert estimation["reference"] == ("ref_round:ch04" if use_external else "round1")
    assert estimation["qc"]["details"]["transform"] == {"kind": "translation", "displacement_zyx": [0, 3, -2]}
    # Another path holds the same record.
    other = fov.save_registration_record(tmp_path / "elsewhere" / "record.json")
    assert other.read_bytes() == path.read_bytes()


def test_save_registration_record_without_other_rounds_is_empty(tmp_path):
    fov, _, _ = other_round_fov(tmp_path)
    record = json.loads(fov.save_registration_record().read_text())
    assert record["registered_rounds"] == {} and record["registration"] == {}


def test_the_removed_writers_are_gone():
    from starfinder.dataset import FOV
    for name in ("save_processing_log", "save_diagnostics", "_save_shift_log"):
        assert not hasattr(FOV, name), name


# --- The final targets of common.smk, per backend ---------------------------------------------------------

RULE_FILES = {"python": ("registration-py.smk", "spot-finding-py.smk"),
              "matlab": ("registration.smk", "spot-finding.smk")}
RULES = ("rsf_single_fov", "gr_single_fov_subtile", "lrsf_single_fov_subtile", "deep_create_subtile",
         "deep_rsf_subtile", "nuclei_registration")


def declared_outputs(backend, rule, names, fovs, subtiles):
    """The output files of ``rule`` in the backend's rule files, expanded over the FOVs and subtiles.

    The output section is evaluated with the names of common.smk (``names``; the Python rules name the FOV folder's
    run record and reference image by variables, W-564) and stand-ins for Snakemake's ``expand`` and ``temp``.
    """
    for name in RULE_FILES[backend]:
        text = (ROOT / "workflow" / "rules" / name).read_text()
        match = re.search(rf"^rule {rule}:\n(.*?)(?=^rule |\Z)", text, flags=re.MULTILINE | re.DOTALL)
        if match:
            break
    else:
        raise AssertionError(f"{rule} is in no {backend} rule file")
    section = re.search(r"^    output:\n(.*?)^    \w+:", match.group(1), flags=re.MULTILINE | re.DOTALL).group(1)

    def expand(pattern, **values):
        values = {key: value if isinstance(value, list) else [value] for key, value in values.items()}
        return [pattern.format(**dict(zip(values, combination))) for combination in itertools.product(*values.values())]
    entries = eval(f"[{section}]", {**names, "expand": expand, "temp": lambda value: value})
    patterns = [str(p) for entry in entries for p in (entry if isinstance(entry, list) else [entry])]
    return {pattern.format(fovID=fov, n_subtile=i) for pattern in patterns for fov in fovs for i in subtiles}


@pytest.mark.workflow
@pytest.mark.parametrize("backend", ["python", "matlab"])
@pytest.mark.parametrize("rule", RULES)
def test_the_final_targets_branch_on_the_backend(tmp_path, monkeypatch, backend, rule):
    config, output_root = repository_config("docs/examples/workflow-minimal.yaml", tmp_path)
    config.update(backend=backend, workflow_mode="free", rules={rule: {"run": True}})
    if backend == "matlab":
        # The MATLAB backend's membership: the sample annotation, fov_id_pattern, n_fovs and the subset keys, for the
        # FOVs of the example's fovs.csv (tile_1 of sample1, tile_2 of sample2).
        config.update(fov_id_pattern="tile_{i}", n_fovs=2, subset_list=[], subset_range=False, subset_random=False)
        annotation(output_root, [(SAMPLE, 1, 1), ("sample2", 2, 2)])
    config.pop("assembly", None)
    config.pop("stitching", None)
    names = common_preamble(config, output_root, monkeypatch)
    assert names["BACKEND"] == backend
    fovs, subtiles = names["FOVS"], names["N_SUBTILE"]
    targets = names["get_overall_output"](None)
    assert targets and len(set(targets)) == len(targets)
    # Every target is an output of the rule in the backend's own rule files.
    # The names the Python block of common.smk adds after its preamble, for the Python rule files.
    names.update(RUN_RECORD=_run_record_outputs(output_root, ["{fovID}"])[0],
                 SUBTILE_RUN_RECORD=_run_record_outputs(output_root, ["{fovID}"], ["{n_subtile}"])[0],
                 REFERENCE_IMAGE_OUTPUT=_reference_image_outputs(config, output_root, ["{fovID}"]))
    assert set(targets) <= declared_outputs(backend, rule, names, fovs, subtiles)
    logs = [t for t in targets if t.startswith(f"{output_root}/log/")]
    if backend == "python":
        assert logs == []
        if rule == "nuclei_registration":
            assert targets == [f"{output_root}/fovs/{fov}/nuclei_registration.json" for fov in fovs]
        elif rule in ("lrsf_single_fov_subtile", "deep_rsf_subtile"):
            assert {f"{output_root}/fovs/{fov}/subtile_{i}/run.json" for fov in fovs for i in subtiles} <= set(
                targets)
        else:
            assert {f"{output_root}/fovs/{fov}/run.json" for fov in fovs} <= set(targets)
    else:
        # MATLAB's rules keep their logs (deep_create_subtile requests none) and write no run.json.
        assert bool(logs) is (rule != "deep_create_subtile")
        assert not any(t.endswith("run.json") or t.endswith("nuclei_registration.json") for t in targets)
