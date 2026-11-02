"""Private one-process method runner: ``python -m starfinder.benchmark._trial trial.json``.

A trial has ``method`` (module, method/config for registration or steps for
preprocessing), ``inputs`` (reference/moving or image), ``seed`` and
``root_output_path``. Input paths are relative to the trial file; each may
instead be a {path, sha256} entry. Optional ``metadata`` describes the common
ImageMetadata grid. NPY inputs are finite package-order arrays; other image
files enter through load_volume. Outputs are NPY images, registration transform
JSON/NPZ or preprocessing fitted records, and record.json. A fresh output folder
is required. This entry runs methods only; it never computes metrics.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import random
import sys
import time

import numpy as np
import yaml

from starfinder._records import file_sha256, jsonable, write_json
from starfinder._resources import process_resources
from starfinder.image import ImageMetadata
from starfinder.workflow._shape import configuration_sha256, plain


def _read(path):
    value = yaml.safe_load(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value


def _config(name, values, registry):
    types = {spec.name: cls for cls, spec in registry.items()}
    if name not in types:
        raise ValueError(f"unknown method {name!r}")
    cls = types[name]
    values = dict(values)
    # Resolved configurations retain the canonical dataclass type and init=False method.
    if values.pop("type", cls.__name__) != cls.__name__:
        raise ValueError("configuration type disagrees with method")
    if "method" not in {f.name for f in fields(cls) if f.init}:
        if values.pop("method", name) != name:
            raise ValueError("configuration method disagrees with method")
    return cls(**values)


def _method(value, seed):
    from starfinder.preprocessing import PREPROCESSING_METHODS, PreprocessingRecipe, PreprocessingStep
    from starfinder.registration import REGISTRATION_METHODS
    value = dict(value)
    module = value.pop("module")
    if module == "registration":
        name, params = value.pop("method"), dict(value.pop("config", {}))
        cls = next((cls for cls, spec in REGISTRATION_METHODS.items() if spec.name == name), None)
        if cls is not None and "random_seed" in {f.name for f in fields(cls)}:
            params.setdefault("random_seed", seed)
        config = _config(name, params, REGISTRATION_METHODS)
        resolved = {"module": module, "method": name, "config": plain(config)}
    elif module == "preprocessing":
        entries = value.pop("steps")
        if not entries:
            raise ValueError("preprocessing trial requires at least one step")
        steps = []
        for entry in entries:
            if set(entry) - {"method", "config"}:
                raise ValueError("unsupported preprocessing step keys")
            steps.append(PreprocessingStep(_config(entry["method"], entry.get("config", {}),
                                                   PREPROCESSING_METHODS)))
        config = PreprocessingRecipe(tuple(steps))
        resolved = {"module": module, "steps": [{"method": PREPROCESSING_METHODS[type(s.config)].name,
                                                 "config": plain(s.config)} for s in config.steps]}
    else:
        raise ValueError(f"unsupported trial module {module!r}")
    if value:
        raise ValueError(f"unsupported method keys: {sorted(value)}")
    return resolved, config


def _resolve(value, base):
    if set(value) - {"method", "inputs", "seed", "root_output_path", "metadata", "condition"}:
        raise ValueError("unsupported trial keys")
    seed = value["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    method, _ = _method(value["method"], seed)
    expected = {"reference", "moving"} if method["module"] == "registration" else {"image"}
    # A preprocessing step that needs a reference may explicitly supply it.
    keys = set(value["inputs"])
    if keys != expected and not (method["module"] == "preprocessing" and keys == {"image", "reference"}):
        raise ValueError(f"inputs must name {sorted(expected)} (preprocessing may also supply reference)")
    inputs = {}
    for role, entry in value["inputs"].items():
        entry = {"path": entry} if isinstance(entry, str) else dict(entry)
        if set(entry) - {"path", "sha256"}:
            raise ValueError("unsupported input keys")
        path = (base / entry["path"]).resolve()
        digest = file_sha256(path)
        if "sha256" in entry and entry["sha256"] != digest:
            raise ValueError(f"input SHA-256 changed: {path}")
        inputs[role] = {"path": str(path), "sha256": digest}
    metadata = plain(ImageMetadata(**value.get("metadata", {"frame_id": "trial/reference"})))
    metadata.pop("type")
    resolved = {"method": method, "inputs": inputs, "metadata": metadata, "seed": seed,
                "condition": value.get("condition", {})}
    if "root_output_path" in value:
        resolved["root_output_path"] = str((base / value["root_output_path"]).resolve())
    return resolved


def _load(entry):
    path = Path(entry["path"])
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"input SHA-256 changed: {path}")
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    from starfinder.io import load_volume
    return load_volume(path).image


def _run(configuration, out):
    method, config = _method(configuration["method"], configuration["seed"])
    inputs = {name: _load(entry) for name, entry in configuration["inputs"].items()}
    metadata = ImageMetadata(**configuration["metadata"])
    paths = []
    if method["module"] == "registration":
        from starfinder.registration import (estimate_transform, apply_transform, BSplineTransform,
                                             DenseDisplacementTransform)
        from starfinder._checkpoint import _transform_json
        result = estimate_transform(inputs["reference"], inputs["moving"], config=config,
                                    reference_metadata=metadata, moving_metadata=metadata)
        image = apply_transform(inputs["moving"], result.transform, config=result.application_config)
        field_name = None
        transform = result.transform
        if isinstance(transform, (BSplineTransform, DenseDisplacementTransform)):
            field_name = "transform.npz"
            array = transform.coefficients if isinstance(transform, BSplineTransform) else transform.displacement_zyx
            np.savez(out / field_name, result_0=array)
            paths.append(out / field_name)
        write_json(jsonable(_transform_json(result, 0, field_name, result.application_config)), out / "transform.json")
        paths.append(out / "transform.json")
    else:
        from starfinder.preprocessing import StepContext, run_step, step_spec
        image = inputs["image"]
        records = []
        for step in config.steps:
            reference = inputs.get("reference") if step_spec(step.config).scope == "needs_reference" else None
            context = StepContext("trial", "reference", metadata, reference=reference)
            result = run_step(image, step.config, context)
            image = result.image
            records.append({"config": plain(step.config), "fitted": result.fitted, "diagnostics": result.diagnostics})
        write_json(jsonable(records), out / "preprocessing.json")
        paths.append(out / "preprocessing.json")
    np.save(out / "image.npy", image, allow_pickle=False)
    paths.append(out / "image.npy")
    return [{"path": p.name, "sha256": file_sha256(p), "bytes": p.stat().st_size} for p in paths]


def _main(path):
    path = Path(path).resolve()
    value = _read(path)
    # Reserve the output before resolution, so method/input failures also have a record.
    out = (path.parent / value["root_output_path"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"trial output must be empty: {out}")
    started = time.perf_counter()
    record = {"format": "starfinder.benchmark.trial/1", "configuration": value,
              "seed": value.get("seed"), "inputs": {}, "outputs": [], "status": "failed", "error": None}
    try:
        configuration = _resolve(value, path.parent)
        record.update(configuration=configuration, inputs=configuration["inputs"],
                      configuration_sha256=configuration_sha256(configuration))
        random.seed(configuration["seed"])
        np.random.seed(configuration["seed"])
        record["outputs"] = _run(configuration, out)
        record["status"] = "succeeded"
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        # Only the private entry's declared files in its freshly reserved folder.
        for name in ("transform.npz", "transform.json", "preprocessing.json", "image.npy"):
            (out / name).unlink(missing_ok=True)
    record.update(process_resources(started))
    write_json(record, out / "record.json")
    return 0 if record["status"] == "succeeded" else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m starfinder.benchmark._trial <trial.json>")
    raise SystemExit(_main(sys.argv[1]))
