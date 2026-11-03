"""Private deterministic methods x conditions x seeds x inputs expansion.

YAML/JSON experiment mappings contain lists ``methods``, ``conditions``,
``seeds`` and ``inputs``. Methods use _trial's method mapping. Each condition
is {name, method, inputs} with optional recursive method/input overrides; each input is a
_trial input-role mapping. Optional common ``metadata`` is passed to trials.
No processing runs during expansion and no engine is imported.
"""
from copy import deepcopy
from itertools import product
from pathlib import Path

from starfinder._records import file_sha256, write_json
from starfinder.workflow._shape import canonical, configuration_sha256, sha256
from ._trial import _read, _resolve


def _merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def expand_experiment(path, output_dir):
    """Write resolved private trial configurations and manifest.json; return the manifest.

    Paths are relative to the experiment file, not the process's directory.
    Trials are ordered by method, condition, seed, input. Hashes use workflow
    canonical JSON/configuration_sha256 and exclude root_output_path, as in
    workflows. Manifest entries include both configuration and file-byte hashes.
    Re-expansion to the same directory produces identical bytes. Missing or
    changed inputs fail resolution before any configuration is written.
    """
    path, output_dir = Path(path).resolve(), Path(output_dir).resolve()
    experiment = _read(path)
    if set(experiment) - {"methods", "conditions", "seeds", "inputs", "metadata", "root_output_path"}:
        raise ValueError("unsupported experiment keys")
    axes = [experiment[key] for key in ("methods", "conditions", "seeds", "inputs")]
    if any(not isinstance(axis, list) or not axis for axis in axes):
        raise ValueError("each experiment axis must be a nonempty list")
    resolved = []
    output_root = (path.parent / experiment.get("root_output_path", str(output_dir))).resolve()
    for i, (method, condition, seed, inputs) in enumerate(product(*axes)):
        if set(condition) - {"name", "method", "inputs"}:
            raise ValueError("condition accepts name, method and input overrides only")
        trial_inputs = _merge(inputs, condition.get("inputs", {}))
        for entry in trial_inputs.values():
            if isinstance(entry, dict) and "scene" in entry:
                entry["scene"].setdefault("condition", condition.get("name"))
                entry["scene"].setdefault("seed", seed)
        trial = {"method": _merge(method, condition.get("method", {})), "inputs": trial_inputs, "seed": seed,
                 "condition": condition, "root_output_path": str(output_root / f"trial-{i:06d}")}
        if "metadata" in experiment:
            trial["metadata"] = experiment["metadata"]
        configuration = _resolve(trial, path.parent)
        resolved.append(configuration)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, configuration in enumerate(resolved):
        name = f"trial-{i:06d}.json"
        data = canonical(configuration)
        (output_dir / name).write_bytes(data)
        entries.append({"path": name, "sha256": sha256(data),
                        "configuration_sha256": configuration_sha256(configuration)})
    manifest = {"format": "starfinder.benchmark.experiment/1",
                "source": {"path": str(path), "sha256": file_sha256(path)}, "trials": entries}
    write_json(manifest, output_dir / "manifest.json")
    return manifest
