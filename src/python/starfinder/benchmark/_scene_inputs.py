"""Private regenerative inputs for the explicitly uncalibrated E13a pilot.

Resolution builds configurations only. Generation happens once per scene in
the method process, with no persisted input image or dependence on job paths.
"""
import math

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


def resolve_scene_input(entry, seed):
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
