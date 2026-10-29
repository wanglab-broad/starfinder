"""The six appearance presets of the real-appearance scene (design/specs/synthetic/real-appearance.md, "Presets", PR1).

One routine and one ``slow`` preset per context, beside the six accepted cellular presets, which stay as they are.
Each takes the accepted cellular preset of its context and tier unchanged (its grid, spacing, jitter, overlap, seed,
cell types, molecules, stains and morphology round) except for its own dataset version and scene key. The routine
presets place the entries of their context's closed-form bank (CF1) by a fixed count (DN3), the count that DN1
reached once on the same configuration. The ``slow`` presets place the entries of the real banks of SRC2, each named
by its name and SHA-256 (ST1, VA7), until the composed stain's visible-stain share reaches DN1's target; a stored
bank holds no source image, so the target and the threshold are given here, measured on the bank's source images by
the build of these presets with the generator's own arithmetic (``_appearance._density_rows``). No bank, record,
source configuration or array derived from real images is kept in the repository.
"""
# Build issue W-519
from __future__ import annotations

from dataclasses import replace

from starfinder.readout import Codebook

from ._appearance import _APPEARANCE_PRESET_VERSION, AppearanceSceneConfig
from ._appearance_bank import BankReference
from ._appearance_layout import LayoutConfig

#: Version of the appearance presets; any change to a preset's value bumps it.
APPEARANCE_PRESET_VERSION = _APPEARANCE_PRESET_VERSION

# The real banks of SRC2 (W-519's builds; the master copies and records under the artifact root's appearance-banks/,
# the working copies in STARFINDER_APPEARANCE_BANKS), by name and the SHA-256 of the bank file.
_REAL_BANKS = {
    "ln-sample1-v1": "d6b07147b44a31913fc3572997ddbab9a024e940167f8a45237357018f5e0573",
    "tissue2d-v1": "6ccdf769a9a92cd3ded253e5e1d24e01ec6603601147daa15761d7e90ab0c02d",
    "culture3d-v1": "a0ee7c13449454bf55e6f2c48e99add84f1f1d261da30f675067fba486a11096",
}
# Values by preset. ``count`` (DN3) is the count DN1 reached on the routine preset with the closed-form bank's own
# target and threshold, checked after every placement (W-519). ``target`` and ``threshold`` (DN1, over the plateau)
# are the medians over the real bank's source images on their middle planes matching the preset's Z extent, and
# ``check_every`` the prototype's (30 in LN, 15 in 2D tissue, 1 in the culture; proto.py::place_all). The dense 3D
# target and threshold are those of W-513's LN rows of the middle 16 planes (tables/visible_share_ln.csv,
# ln/<FOV>/slab16_z17/DAPI-3D/nuclei-3D): the preset's sample grid holds 16 planes.
# The culture slow preset stops after 500 refused proposals in a row instead of the prototype's 4,000
# (design/specs/synthetic/real-appearance.md, "Presets", the amendment of 2026-10-17): whole cell masks saturate below its
# target, so the stop by refusals ends its placement, and 4,000 refusals took more than 15 minutes on one CPU.
_PRESET_SPECS = {
    "appearance-dense3d-v1": dict(
        context="dense3d", tier="routine", accepted="context-dense3d-v1", bank=None, count=10),
    "appearance-tissue2d-v1": dict(
        context="tissue2d", tier="routine", accepted="context-tissue2d-v1", bank=None, count=15),
    "appearance-culture-v1": dict(
        context="culture", tier="routine", accepted="context-culture-v1", bank=None, count=10),
    "appearance-dense3d-slow-v1": dict(
        context="dense3d", tier="slow", accepted="context-dense3d-slow-v1", bank="ln-sample1-v1",
        target=0.5386069067424004, threshold=0.5672601980832883, check_every=30),
    "appearance-tissue2d-slow-v1": dict(
        context="tissue2d", tier="slow", accepted="context-tissue2d-slow-v1", bank="tissue2d-v1",
        target=0.32425856590270996, threshold=0.5574244417149277, check_every=15),
    "appearance-culture-slow-v1": dict(
        context="culture", tier="slow", accepted="context-culture-slow-v1", bank="culture3d-v1",
        target=0.5010272491349481, threshold=0.6937363480199742, check_every=1, max_refusals=500),
}
# The closed-form banks of the routine presets render n scenes of the accepted routine preset (the last held out).
_CLOSED_FORM_N = 4


def _entry(name):
    from ._context_presets import CELLULAR_PRESETS
    spec = _PRESET_SPECS[name]
    accepted = CELLULAR_PRESETS[spec["accepted"]]
    stored = spec["bank"] is not None
    return dict(
        version=APPEARANCE_PRESET_VERSION, seed=accepted["seed"], split=accepted["split"], scene_key=name,
        codebook=accepted["codebook"], context=spec["context"], tier=spec["tier"], accepted_preset=spec["accepted"],
        **{k: accepted[k] for k in ("columns", "rows", "fov_shape_zyx", "spacing_zyx", "overlap", "jitter_max_zyx")},
        bank=spec["bank"] if stored else f"closed-form {spec['accepted']}",
        bank_sha256=_REAL_BANKS[spec["bank"]] if stored else None,
        bank_preset=None if stored else spec["accepted"], bank_n=None if stored else _CLOSED_FORM_N,
        density="visible" if stored else "count", count=None if stored else spec["count"],
        target=spec.get("target"), threshold=spec.get("threshold"), check_every=spec.get("check_every", 1),
        max_refusals=spec.get("max_refusals", LayoutConfig().max_refusals))


#: Appearance presets (appearance-presets-v1), separate from CELLULAR_PRESETS.
APPEARANCE_PRESETS = {name: _entry(name) for name in _PRESET_SPECS}


def appearance_scene_preset(name: str) -> tuple[Codebook, AppearanceSceneConfig]:
    """Return (codebook, config) for an APPEARANCE_PRESETS name: an ``AppearanceSceneConfig``.

    The six presets of design/specs/synthetic/real-appearance.md ("Presets", PR1). The routine presets
    ``appearance-dense3d-v1``, ``appearance-tissue2d-v1`` and ``appearance-culture-v1``
    place a fixed count of entries (DN3) of the closed-form bank of their context's routine
    preset (``BankReference(preset=..., n=4)``); they read no real data. The ``slow`` presets
    ``appearance-dense3d-slow-v1``, ``appearance-tissue2d-slow-v1`` and
    ``appearance-culture-slow-v1`` place entries of the real banks ``ln-sample1-v1``,
    ``tissue2d-v1`` and ``culture3d-v1``, named by name and SHA-256 and read from the
    directory that ``STARFINDER_APPEARANCE_BANKS`` names, until the visible-stain share
    reaches its target (DN1) or placement stops by its refusals in a row (500 for the culture,
    whose cells saturate below its target; 4,000 otherwise). Each preset's context is the
    accepted cellular preset of its context and tier (``cellular_scene_preset``) with the
    preset's name as its dataset version
    and scene key; augmentation and composition take their defaults (AU1; H2, BG1 for a real
    bank and BG2 for a closed-form bank, no noise, the bank's median plateau). Unknown names
    raise ValueError.
    """
    from ._context_presets import cellular_scene_preset
    if name not in _PRESET_SPECS:
        raise ValueError(f"unknown appearance preset: {name}; choose from {list(_PRESET_SPECS)}")
    entry = APPEARANCE_PRESETS[name]
    codebook, context = cellular_scene_preset(entry["accepted_preset"])
    fov = replace(context.sample.fov, dataset_version=name, scene_key=name)
    context = replace(context, sample=replace(context.sample, fov=fov))
    if entry["bank_sha256"] is None:
        bank = BankReference(preset=entry["bank_preset"], n=entry["bank_n"])
        layout = LayoutConfig(density="count", count=entry["count"], max_refusals=entry["max_refusals"])
    else:
        bank = BankReference(name=entry["bank"], sha256=entry["bank_sha256"])
        layout = LayoutConfig(density="visible", target=entry["target"], threshold=entry["threshold"],
                              check_every=entry["check_every"], max_refusals=entry["max_refusals"])
    return codebook, AppearanceSceneConfig(context=context, bank=bank, layout=layout)
