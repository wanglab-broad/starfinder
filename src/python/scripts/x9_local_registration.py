"""X9: direct versus locally registered tiles on thick_medium/gaussian_small.

Run on CPU 8 under /usr/bin/time -v with an explicit --rss-limit-gib grant.
The watchdog stops at the granted RSS limit,
three hours or less than 15 GB free on root. Transient local scratch is
removed at exit; only molecule tables and measurements are retained.
--smoke exercises the same execution branch on the small exactness fixture,
and is explicitly not an X9 result. X6's thresholds are never changed.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SHAPE = (100, 1024, 1024)


def _json_default(value):
    """Scene configuration contains NumPy arrays as well as JSON scalar fields."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n")
    if os.environ.get("STEP_WRITES"):
        with Path(os.environ["STEP_WRITES"]).open("a") as handle:
            handle.write(str(path) + "\n")


def _worker(args):
    """One stage per process so each run's maximum RSS is independent."""
    faulthandler.enable()
    faulthandler.dump_traceback_later(300, repeat=True)
    from test import tiled_fixtures as F
    from test.test_workflow_tiled import _matched, _table, run_direct, run_tiled
    from starfinder import _layout
    root = args.scratch / "input"
    name = "exactness-deformed" if args.smoke else "thick-medium-deformed"
    start = time.monotonic()
    if args.worker == "generate":
        from dataclasses import replace
        from starfinder.synthetic import benchmark_scene_preset, generate_amplicon_dataset
        from starfinder.synthetic._presets import _appearance, deformation_geometry
        from starfinder.synthetic._storage import _write_channels, _write_truth, _amplicon_generation
        preset = "small" if args.smoke else "thick_medium"
        book, config = benchmark_scene_preset(preset)
        if args.smoke:
            config = replace(config, shape_zyx=F.SHAPE, count=F.COUNT, max_count=1024,
                             dataset_version="w644-exactness-v1", scene_key="w644/exactness",
                             **_appearance(F.SHAPE, len(book.round_labels), "uint16"))
        else:
            assert config.shape_zyx == SHAPE
        config = replace(config, geometry=deformation_geometry(
            "gaussian_small", config.shape_zyx, reference_round=book.round_labels[0],
            translation_max_zyx=config.geometry.translation_max_zyx))
        folder = root / name / F.SAMPLE
        def write(fov, label, image, metadata):
            _write_channels(folder / label / fov, image, book.channel_labels)
        dataset = generate_amplicon_dataset(book, config, fov_ids=(F.FOV,), preset=preset, on_round=write)
        table = book.table
        table.assign(entry_id=table.base_sequence)[["entry_id", "gene_id", "color_sequence"]].to_csv(
            folder / "codebook.csv", index=False)
        _write_truth(folder, dataset, _amplicon_generation(preset=preset, seed=config.seed,
                     dtype=config.dtype, noise=True, fov_ids=(F.FOV,)))
        def identity(path):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}
        result = {"seed": config.seed,
                  "inputs": [identity(p) for p in sorted(root.rglob("*")) if p.is_file()],
                  "processing": F.configuration(root, args.scratch / "direct", name)}
        from starfinder._records import jsonable
        result["scene_configuration"] = jsonable(config)
    else:
        size = None if args.worker == "direct" else int(args.worker)
        output = args.scratch / args.worker
        if size is None:
            folder, count = run_direct(root, output, name, local=True), 1
        else:
            folder, count = run_tiled(root, output, name, size, 12, local=True)
        source = folder / _layout.SPOT_TABLES["molecules"]
        target = args.output.parent / (args.worker + "-molecules.csv")
        shutil.copyfile(source, target)
        table = _table(target)
        assert len(table) > 0
        direct = _table(args.output.parent / "direct-molecules.csv")
        found = _matched(direct, table) / len(direct)
        kept = _matched(table, direct) / len(table)
        difference = abs(len(table) - len(direct)) / len(direct)
        result = {"tile_size": size, "tile_count": count, "direct_count": len(direct), "count": len(table),
                  "direct_matched": found, "tiled_matched": kept, "relative_count_difference": difference,
                  "table": str(target), "table_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                  "passed": found >= .95 and kept >= .95 and difference <= .05}
        if os.environ.get("STEP_WRITES"):
            with Path(os.environ["STEP_WRITES"]).open("a") as handle:
                handle.write(str(target) + "\n")
        # Earlier grids' checkpoints are transient and need not coexist.
        shutil.rmtree(output)
    result.update(wall_seconds=time.monotonic() - start, maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    _save(args.scratch / (args.worker + ".json"), result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--rss-limit-gib", type=int, required=True,
                        help="Granted process RSS limit in GiB (required; no default)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", choices=("generate", "direct", "512", "256", "32", "22"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rss_limit_gib <= 0:
        parser.error("--rss-limit-gib must be positive")
    if args.worker:
        return _worker(args)
    temporary_root = Path(os.environ["TMPDIR"]).resolve()
    if not args.scratch.resolve().is_relative_to(temporary_root) or sorted(os.sched_getaffinity(0)) != [8]:
        parser.error("scratch must be under local TMPDIR and execution pinned to CPU 8")
    if args.scratch.exists():
        parser.error("scratch must not already exist")
    record = {"check": "X9-smoke" if args.smoke else "X9", "fixture": "exactness-deformed" if args.smoke else "thick_medium",
              "shape_zyx": (16, 64, 64) if args.smoke else SHAPE, "deformation": "gaussian_small", "halo": 12,
              "seed": 42, "actual_affinity": sorted(os.sched_getaffinity(0)),
              "initial_root_free_bytes": shutil.disk_usage("/").free,
              "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
              "rss_poll_seconds": 1 if args.smoke else 10, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "commands": [], "host": platform.node(), "status": "running", "measurements": [],
              "grant": {"affinity": [8], "cpu": 1, "rss_kib": args.rss_limit_gib * 1024**2, "wall_seconds": 1800 if args.smoke else 10800},
              "scratch": str(args.scratch), "peak_scratch_bytes": 0, "scratch_deleted": False,
              "against_x6": {"direct_matched_minimum": .95, "tiled_matched_minimum": .95,
                             "relative_count_difference_maximum": .05, "conclusion": "Not yet measured."}}
    import psutil
    start = time.monotonic()
    rss_limit = args.rss_limit_gib * 1024**3
    wall_limit = 1800 if args.smoke else 10800
    args.scratch.mkdir(parents=True)
    _save(args.output, record)
    process = None
    try:
        for stage in ("generate", "direct", *(('32', '22') if args.smoke else ('512', '256'))):
            command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output), "--scratch", str(args.scratch), "--rss-limit-gib", str(args.rss_limit_gib), "--worker", stage]
            if args.smoke:
                command.append("--smoke")
            record["commands"].append(command)
            log = args.output.parent / (stage + ".log")
            with log.open("w") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                reason = None
                sampled_rss = 0
                while process.poll() is None:
                    processes = [psutil.Process(process.pid)]
                    try:
                        processes += processes[0].children(recursive=True)
                        rss = sum(p.memory_info().rss for p in processes)
                    except psutil.NoSuchProcess:
                        rss = 0
                    sampled_rss = max(sampled_rss, rss)
                    scratch_bytes = 0
                    for path in args.scratch.rglob("*"):
                        try:
                            if path.is_file():
                                scratch_bytes += path.stat().st_size
                        except FileNotFoundError:
                            pass  # A worker can remove a completed grid during the sample.
                    record["peak_scratch_bytes"] = max(record["peak_scratch_bytes"], scratch_bytes)
                    if "zarr/core/sync.py" in log.read_text():
                        reason = "Sandbox Zarr sync hang; stack retained in " + str(log)
                    elif rss > rss_limit:
                        reason = f"{rss_limit // 1024**3} GiB RSS limit exceeded"
                    elif time.monotonic() - start > wall_limit:
                        reason = "3 h wall limit exceeded"
                    elif shutil.disk_usage("/").free < 15_000_000_000:
                        reason = "Root free disk below 15 GB"
                    if reason:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        break
                    time.sleep(1 if args.smoke else 10)
            if os.environ.get("STEP_WRITES"):
                with Path(os.environ["STEP_WRITES"]).open("a") as handle:
                    handle.write(str(log) + "\n")
            if reason:
                record.update(status="not_run", attempt_status="not_run" if reason.startswith("Sandbox") else "resource_stop",
                              reason=reason, stopped_stage=stage, stopped_stage_sampled_rss_kib=sampled_rss // 1024)
                break
            if process.returncode:
                record.update(status="failed", reason=f"Execution error in {stage}; see {log}")
                raise RuntimeError(record["reason"])
            measured = json.loads((args.scratch / (stage + ".json")).read_text())
            if stage == "generate":
                record["generation"] = measured
                record["seed"] = measured["seed"]
            else:
                record["measurements"].append(measured)
            _save(args.output, record)
            if measured["maximum_rss_kib"] * 1024 > rss_limit:
                record.update(status="not_run", attempt_status="resource_stop", reason=f"{rss_limit // 1024**3} GiB RSS limit exceeded in measured maximum", stopped_stage=stage)
                break
        else:
            record["status"] = "passed" if all(m["passed"] for m in record["measurements"]) else "failed"
            record["against_x6"]["conclusion"] = (
                "Execution smoke only; full-size X9 proportions remain unverified." if args.smoke else
                "Confirmed." if record["status"] == "passed" else
                "Not confirmed; measured proportions replace the full-size estimate without changing X6.")
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        shutil.rmtree(args.scratch)
        record.update(scratch_deleted=True, wall_seconds=time.monotonic() - start)
        _save(args.output, record)
    print(record["status"])


if __name__ == "__main__":
    main()
