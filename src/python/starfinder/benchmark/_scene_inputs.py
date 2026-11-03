"""Private regenerative trial inputs: the uncalibrated amplicon scene and the cellular presets.

The ``uncalibrated_preprocessing`` amplicon scene (the explicitly uncalibrated
E13a pilot) feeds preprocessing and spot finding by ``round`` and optional
``channel``. A ``CELLULAR_PRESETS`` scene feeds segmentation by ``fov``,
``round`` (``reference`` or ``morphology``) and ordered ``stains``.
Resolution builds configurations only. Generation happens once per scene in
the method process, with no persisted input image or dependence on job paths.
"""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
import math

import numpy as np

from starfinder.workflow._shape import configuration_sha256, plain


def _configuration(identity, seed):
    from starfinder.synthetic import (UNCALIBRATED_PREPROCESSING_VERSION,
                                      uncalibrated_preprocessing_scene_config)
    identity = dict(identity)
    if set(identity) - {"preset", "version", "condition", "seed", "k"}:
        raise ValueError("unsupported scene identity keys")
    if identity.get("preset") != "uncalibrated_preprocessing":
        raise ValueError("unsupported scene preset")
    if identity.get("version") != UNCALIBRATED_PREPROCESSING_VERSION:
        raise ValueError("scene preset version disagrees with installed version")
    scene_seed = identity.get("seed", seed)
    if isinstance(scene_seed, bool) or not isinstance(scene_seed, int) or scene_seed != seed:
        raise ValueError("scene seed must equal the trial seed")
    identity["seed"] = seed
    if "k" in identity and (identity.get("condition") != "saturation" or isinstance(identity["k"], bool)
                            or not isinstance(identity["k"], (int, float))
                            or not math.isfinite(identity["k"]) or identity["k"] <= 0):
        raise ValueError("k must be a finite positive saturation multiplier")
    book, config = uncalibrated_preprocessing_scene_config(identity["condition"], seed=seed, k=identity.get("k"))
    return identity, book, config


def _plain_arrays(value):
    """plain() of a configuration that also holds NumPy arrays (the cellular presets), arrays as nested lists."""
    if is_dataclass(value) and not isinstance(value, type):
        return {"type": type(value).__name__, **{f.name: _plain_arrays(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, Mapping):
        return {str(k): _plain_arrays(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_arrays(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    return plain(value)


def _cellular(identity):
    from starfinder.synthetic import CELLULAR_PRESETS
    return isinstance(identity, dict) and identity.get("preset") in CELLULAR_PRESETS


def _cellular_configuration(identity, seed):
    from starfinder.synthetic import CELLULAR_PRESET_VERSION, cellular_scene_preset
    identity = dict(identity)
    # condition is a label the experiment expansion adds; the presets have no conditions.
    if set(identity) - {"preset", "version", "condition", "seed"}:
        raise ValueError("unsupported scene identity keys")
    if identity.get("version") != CELLULAR_PRESET_VERSION:
        raise ValueError("scene preset version disagrees with installed version")
    scene_seed = identity.get("seed", seed)
    if isinstance(scene_seed, bool) or not isinstance(scene_seed, int) or scene_seed != seed:
        raise ValueError("scene seed must equal the trial seed")
    identity["seed"] = seed
    book, config = cellular_scene_preset(identity["preset"])
    sample = config.sample
    config = replace(config, sample=replace(sample, fov=replace(sample.fov, seed=seed)))
    return identity, book, config


def _resolve_cellular(entry, seed):
    if set(entry) - {"scene", "fov", "round", "stains", "scene_configuration_sha256"}:
        raise ValueError("unsupported scene input keys")
    identity, _, config = _cellular_configuration(entry["scene"], seed)
    if entry.get("fov") not in config.sample.acquisition.ids():
        raise ValueError("scene input fov is not in the preset")
    round_name = entry.get("round", "reference")
    if round_name not in ("reference", "morphology"):
        raise ValueError("cellular scene input round must be 'reference' or 'morphology'")
    names = [stain.name for stain in config.stains if stain.round == round_name]
    stains = entry.get("stains")
    if (not isinstance(stains, list) or not stains or len(set(stains)) != len(stains)
            or any(name not in names for name in stains)):
        raise ValueError(f"scene input stains must name distinct {round_name} stains of {names}")
    digest = configuration_sha256(_plain_arrays(config))
    if entry.get("scene_configuration_sha256", digest) != digest:
        raise ValueError("scene configuration SHA-256 changed")
    return {"scene": identity, "fov": entry["fov"], "round": round_name, "stains": list(stains),
            "scene_configuration_sha256": digest}


def resolve_scene_input(entry, seed, module="preprocessing"):
    """Resolve one scene input for a trial module; segmentation takes cellular scenes, the others amplicon."""
    if (module == "segmentation") != _cellular(entry.get("scene")):
        raise ValueError(f"unsupported scene preset for {module}")
    if module == "segmentation":
        return _resolve_cellular(entry, seed)
    if set(entry) - {"scene", "round", "channel", "scene_configuration_sha256"}:
        raise ValueError("unsupported scene input keys")
    identity, book, config = _configuration(entry["scene"], seed)
    if entry.get("round") not in book.round_labels:
        raise ValueError("scene input round is not in the codebook")
    if "channel" in entry and entry["channel"] not in book.channel_labels:
        raise ValueError("scene input channel is not in the codebook")
    digest = configuration_sha256(plain(config))
    if entry.get("scene_configuration_sha256", digest) != digest:
        raise ValueError("scene configuration SHA-256 changed")
    resolved = {"scene": identity, "round": entry["round"], "scene_configuration_sha256": digest}
    if "channel" in entry:
        resolved["channel"] = entry["channel"]
    return resolved


def load_scene_input(entry, cache):
    if _cellular(entry["scene"]):
        return _load_cellular(entry, cache)
    from starfinder.synthetic import generate_amplicon_scene
    entry = resolve_scene_input(entry, entry["scene"]["seed"])
    key = entry["scene_configuration_sha256"]
    if key not in cache:
        _, book, config = _configuration(entry["scene"], entry["scene"]["seed"])
        cache[key] = generate_amplicon_scene(book, config=config)
    scene = cache[key]
    image = scene.rounds[entry["round"]]
    if "channel" in entry:
        image = image[..., scene.channel_labels.index(entry["channel"])]
    return image


def _load_cellular(entry, cache):
    """The ZYXC stack of the named stains, in the named order, of one FOV of a cellular scene."""
    from starfinder.synthetic import generate_cellular_scene
    entry = _resolve_cellular(entry, entry["scene"]["seed"])
    key = entry["scene_configuration_sha256"]
    _, book, config = _cellular_configuration(entry["scene"], entry["scene"]["seed"])
    if key not in cache:
        cache[key] = generate_cellular_scene(book, config=config)
    scene = cache[key]
    if entry["round"] == "reference":
        return np.stack([scene.stains[entry["fov"]][name] for name in entry["stains"]], axis=-1)
    names = [stain.name for stain in config.stains if stain.round == "morphology"]
    return scene.morphology[entry["fov"]][..., [names.index(name) for name in entry["stains"]]]


def scene_hashes(scene):
    """Generated image hashes of a scene for scenes.json; cellular images use the segmentation array_sha256."""
    from starfinder.segmentation._labels import array_sha256
    if hasattr(scene, "provenance"):
        return scene.provenance["image_sha256"]
    hashes = {f"{fov}/reference/{name}": array_sha256(image)
              for fov, stains in scene.stains.items() for name, image in stains.items()}
    hashes.update({f"{fov}/morphology": array_sha256(image) for fov, image in scene.morphology.items()})
    return hashes
