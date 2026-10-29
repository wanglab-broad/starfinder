"""The ``slow`` tier of the cellular chain checks CD, CT and CU as an evidence script (W-603).

These checks of design/validation/synthetic-cellular-scene.md ("Chain checks", as
design/validation/synthetic-real-appearance-scene.md amends them) ran in the controller's ``learned`` chain selections and
in ``pytest-slow-other`` until W-603 moved them to the Chapter II evidence project. They run the whole chain on the
appearance ``slow`` presets, whose banks are built from real images: about 2 h on one CPU and up to 13 GiB per step
(W-542's F5 measurement), so they are evidence workloads rather than software checks. The suite keeps the routine tier
of the same tests, on the closed-form banks.

The cases are the tests of ``src/python/test/test_integration_cellular_chain.py`` themselves, called with the tier
``"slow"`` (and the culture's seeds 42 and 43) on one ``Chains`` of this script: the same assertions, printed rows,
tables, thresholds and per-step measurement as the 31 ``slow`` parameters had at ``ORIGIN_REVISION``. Each context's
chain is built once, on the first case that needs it, and reused by the later cases, as the module fixture did.

    cd src/python
    STARFINDER_APPEARANCE_BANKS=<bank directory> STARFINDER_WEIGHTS_DIR=<weights> \\
        STARFINDER_STARDIST_3D_SPLEEN=<3D_spleen folder> uv run python ../../benchmarks/cellular_chain_evidence.py \\
        --output <run dir>/chain
    uv run python ../../benchmarks/cellular_chain_evidence.py --list

Checks, the rows of the chain tables (``--check`` selects some, or a context: ``dense3d``, ``tissue2d``,
``culture``): ``CD0`` to ``CD5``, ``CT0`` to ``CT5`` and ``CU0`` to ``CU5``. A row's cases run in the order the
tests ran; a case whose chain stage failed earlier fails at once, so that no step runs twice.

Each step (generation, stitch, segmentation, assignment and export, report) runs in its own process under
/usr/bin/time -v with one numerical thread and no GPU. ``--output`` (new or empty, outside the repository) receives
the scenes and exports under ``chains/``, the per-object tables and step logs under ``tables/`` (or where
``STARFINDER_CHAIN_OUTPUT`` names), and ``results.json``: the revision and its dirty state, the bank and model
identities, the seeds, each step's wall time, peak RSS and exit status, and each case's outcome, wall time and the
node ID it was. Exit status 0 when every selected case passed, 1 when one failed, 2 when none failed but one was
skipped (a missing extra, model, bank or /usr/bin/time, each named). ``--list`` and ``--help`` read no bank.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "python"
# The revision whose test suite last held these cases as tests (W-603's starting revision).
ORIGIN_REVISION = "6e243f3123d9aacc704c9062e04b462f358bc394"
ORIGIN_MODULE = "src/python/test/test_integration_cellular_chain.py"
CONTEXTS = {"CD": "dense3d", "CT": "tissue2d", "CU": "culture"}
CHECKS = tuple(f"{prefix}{step}" for prefix in CONTEXTS for step in range(6))
# The 31 moved tests in the order they ran: (row, the test, the culture seed or None). Every case runs with the tier
# "slow"; its node ID is the test's with the parameter [slow] or [slow-<seed>].
CASES = (
    ("CD0", "test_cd0_placement", None),
    ("CT0", "test_ct0_stitch_on_the_projection_sample", None),
    ("CU0", "test_cu0_registration_and_placement", None),
    ("CD1", "test_cd1_nuclei_against_truth", None),
    ("CT1", "test_ct1_nuclei_against_truth", None),
    ("CU1", "test_cu1_nuclei_against_truth", None),
    ("CD2", "test_cd2_cell_of_each_molecule", None),
    ("CT2", "test_ct2_cell_of_each_molecule", None),
    ("CU2", "test_cu2_cell_of_each_molecule", None),
    ("CD2", "test_cd2_truth_labels_ee1", None),
    ("CU2", "test_cu2_truth_labels_ee1", None),
    ("CT2", "test_ct2_truth_labels_sa5", None),
    ("CD3", "test_cd3_count_identities", None),
    ("CT3", "test_ct3_count_identities", None),
    ("CU3", "test_cu3_count_identities", 42),
    ("CU3", "test_cu3_count_identities", 43),
    ("CD3", "test_cd3_count_identities_truth_labels", None),
    ("CT3", "test_ct3_count_identities_truth_labels", None),
    ("CU3", "test_cu3_count_identities_truth_labels", 42),
    ("CU3", "test_cu3_count_identities_truth_labels", 43),
    ("CD4", "test_cd4_report", None),
    ("CT4", "test_ct4_report", None),
    ("CU4", "test_cu4_report", 42),
    ("CU4", "test_cu4_report", 43),
    ("CD4", "test_cd4_report_truth_labels", None),
    ("CT4", "test_ct4_report_truth_labels", None),
    ("CU4", "test_cu4_report_truth_labels", 42),
    ("CU4", "test_cu4_report_truth_labels", 43),
    ("CD5", "test_cd5_scene_as_the_method_sees_it", None),
    ("CT5", "test_ct5_scene_as_the_method_sees_it", None),
    ("CU5", "test_cu5_scene_as_the_method_sees_it", None),
)


def case_name(test, seed):
    return test.removeprefix("test_") + ("" if seed is None else f"-seed{seed}")


def node_id(test, seed):
    """The node ID the case had, as pytest names it from src/python."""
    return f"test/test_integration_cellular_chain.py::{test}[slow{'' if seed is None else f'-{seed}'}]"


def parse(argv):
    parser = argparse.ArgumentParser(
        description="Run the slow tier of the chain checks CD, CT and CU of design/validation/synthetic-cellular-scene.md "
                    "on the real banks, moved out of the test suite by W-603. Each step runs in its own process "
                    "under /usr/bin/time -v; W-542 measured about 2 h on one CPU and up to 13 GiB.",
        epilog="Run from src/python: uv run python ../../benchmarks/cellular_chain_evidence.py --output <run dir>/chain")
    parser.add_argument("--output", type=Path, help="new or empty directory outside the repository for the scenes, "
                                                     "the tables and results.json")
    parser.add_argument("--check", action="append", choices=CHECKS + tuple(CONTEXTS.values()),
                        help="run only this row, or this context's rows (repeatable); default: all, in the order "
                             "the tests ran")
    parser.add_argument("--list", action="store_true",
                        help="list the cases with their former node IDs, the banks and the seeds; reads no bank")
    args = parser.parse_args(argv)
    if not args.list and args.output is None:
        parser.error("--output is required unless --list is given")
    return args


def _chain_module():
    """The test module, imported from src/python/test (the package ``test`` of this checkout, not the standard
    library's); its steps run in processes that import it the same way."""
    if str(PACKAGE) not in sys.path[:1]:
        sys.path.insert(0, str(PACKAGE))
    from test import test_integration_cellular_chain as chain
    return chain


def _revision():
    spec = importlib.util.spec_from_file_location("benchmark_revision", Path(__file__).resolve().parent / "revision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.revision()


def selected_cases(chain, checks=None):
    """The cases (row, test, seed), those of ``checks`` only if given; each test exists in the module and each seed is
    one of its context's ``slow`` seeds there."""
    out = []
    for row, test, seed in CASES:
        assert callable(getattr(chain, test, None)), f"{ORIGIN_MODULE} has no test {test}"
        assert set(case_seeds(chain, row, seed)) <= set(chain.CONTEXTS[CONTEXTS[row[:2]]]["seeds"]["slow"]), test
        if checks is None or row in checks or CONTEXTS[row[:2]] in checks:
            out.append((row, test, seed))
    return out


def case_seeds(chain, row, seed):
    """The seeds of the scenes a case reads: a culture case's own seed, else every ``slow`` seed of its context (the
    culture's steps 0, 1, 2 and 5 pool the seeds 42 and 43)."""
    return [seed] if seed is not None else list(chain.CONTEXTS[CONTEXTS[row[:2]]]["seeds"]["slow"])


def identities(chain):
    """Per context its slow preset, the bank by name and the SHA-256 the preset names, the seeds and the models, from
    the module's ``CONTEXTS`` and ``APPEARANCE_PRESETS``; no bank file is opened."""
    from starfinder.synthetic import APPEARANCE_PRESETS
    out = {}
    for context, spec in chain.CONTEXTS.items():
        preset = spec["slow"]
        out[context] = dict(preset=preset, bank=spec["bank"], bank_sha256=APPEARANCE_PRESETS[preset]["bank_sha256"],
                            seeds=list(spec["seeds"]["slow"]), route=spec["route"], stain=spec["stain"],
                            models=list(spec["models"]))
    return out


def read_identities(chain):
    """The bank and model digests the cases computed from the files they read (``checked_bank``, ``checked_model``)."""
    from starfinder.synthetic._appearance_bank import BANK_DIRECTORY_VARIABLE
    banks = {key.removeprefix("bank:"): dict(path=str(Path(os.environ[BANK_DIRECTORY_VARIABLE]) /
                                                      f"{key.removeprefix('bank:')}.npz"), sha256=digest)
             for key, digest in chain._CHECKED.items() if key.startswith("bank:")}
    models = {key: dict(path=str(chain._model_path(key)), sha256=digests)
              for key, digests in chain._CHECKED.items() if not key.startswith("bank:")}
    return banks, models


def write_results(output, chain, chains, results):
    banks, models = read_identities(chain)
    (output / "results.json").write_text(json.dumps(dict(
        script="benchmarks/cellular_chain_evidence.py", software=_revision(), python=sys.version.split()[0],
        origin=dict(revision=ORIGIN_REVISION, module=ORIGIN_MODULE), contexts=identities(chain),
        banks_read=banks, models_read=models,
        thresholds=dict(TH1=chain.TH1_MATCHED, TH2=chain.TH2_MERGED, TH3=chain.TH3_SPLITS, TH4=chain.TH4_AGREEMENT,
                        TH5_um=chain.TH5_ERROR_UM, amended=sorted("-".join(a) for a in chain.AMENDED)),
        tables=str(chains.runner.output), steps=chains.runner.rows, cases=results),
        indent=1, default=float) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse(argv)
    chain = _chain_module()
    selected = selected_cases(chain, args.check)
    if args.list:
        print(json.dumps(dict(
            contexts=identities(chain), origin=dict(revision=ORIGIN_REVISION, module=ORIGIN_MODULE),
            cases=[dict(case=case_name(t, s), check=r, context=CONTEXTS[r[:2]], seeds=case_seeds(chain, r, s),
                        node_id=node_id(t, s)) for r, t, s in selected]), indent=1))
        return 0
    if not __debug__:
        sys.exit("the cases assert with assert statements, which python -O drops")
    import pytest
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        sys.exit(f"--output {output} is inside the repository {ROOT}")
    if output.exists() and any(output.iterdir()):
        sys.exit(f"--output {output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(chain.OUTPUT, str(output / "tables"))
    chains = chain.Chains(output / "chains")
    results = []
    for row, test, seed in selected:
        name = case_name(test, seed)
        start = time.perf_counter()
        try:
            getattr(chain, test)(chains, "slow", *(() if seed is None else (seed,)))
            outcome, message = "passed", ""
        except AssertionError as error:
            outcome, message = "failed", str(error)
        except pytest.fail.Exception as failure:
            outcome, message = "failed", str(failure.msg)
        except pytest.skip.Exception as skip:
            outcome, message = "skipped", str(skip.msg)
        except Exception as error:  # a case that raises is recorded and the next one runs
            outcome, message = "error", f"{type(error).__name__}: {error}"
        seconds = time.perf_counter() - start
        print(f"{row} {name} ({node_id(test, seed)}) {outcome.upper()} in {seconds:.2f} s"
              + (f": {message}" if message else ""), flush=True)
        results.append(dict(case=name, check=row, context=CONTEXTS[row[:2]], seeds=case_seeds(chain, row, seed),
                            origin_node_id=node_id(test, seed), outcome=outcome, message=message, wall_s=seconds))
        write_results(output, chain, chains, results)
    outcomes = {r["outcome"] for r in results}
    print(f"{len(results)} cases: " + ", ".join(f"{sum(r['outcome'] == o for r in results)} {o}"
                                                 for o in ("passed", "failed", "error", "skipped")))
    return 1 if outcomes & {"failed", "error"} else 2 if "skipped" in outcomes else 0


if __name__ == "__main__":
    sys.exit(main())
