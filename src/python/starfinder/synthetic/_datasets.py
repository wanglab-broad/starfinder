"""Multi-FOV datasets and registration pairs built from formed scenes."""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Callable

import numpy as np
import pandas as pd

from starfinder.barcode import Codebook
from starfinder.image import ImageMetadata

from ._common import _json, _label
from ._formed import FormedSceneConfig, _generate
from ._observation import NoiseConfig
from ._geometry import _blocks, _forward
from ._presets import PRESET_VERSION, _with_seed, registration_scene_preset

#: Version of the derived ground_truth.json layout (historical v2 keys).
HISTORICAL_TRUTH_VERSION = "2.0"


@dataclass
class SyntheticDataset:
    """Formed scenes for several FOVs, their truth tables and provenance.

    ``rounds[fov][round_label]`` holds ZYXC arrays in codebook order, or is
    empty for every FOV when images were handed to ``on_round``. ``metadata``
    has the same keys. ``formed``/``round_truth`` concatenate every FOV (the
    ``namespace`` column names the FOV) and ``spot_truth`` is the per-round,
    per-codeword-channel historical view. ``provenance[fov]`` is the scene
    provenance. ``historical_truth`` is the ground_truth.json payload derived
    from ``formed`` (reference positions) and each round's recorded transform
    (translation and deformation kind). No molecular (biological RNA) truth is
    implied.
    """

    rounds: dict[str, dict[str, np.ndarray]]
    metadata: dict[str, dict[str, ImageMetadata]]
    codebook: Codebook
    formed: pd.DataFrame
    round_truth: pd.DataFrame
    spot_truth: pd.DataFrame
    provenance: dict[str, dict]
    historical_truth: dict

    @property
    def channel_labels(self) -> tuple[str, ...]:
        """Channel labels of every round image, from the codebook."""
        return self.codebook.channel_labels


def _spot_truth(scene, fov):
    """Per-round rows with each amplicon's codeword channel (historical scene_truth)."""
    table = scene.round_truth.merge(scene.formed[["amplicon_id", "gene_id", "codeword", "A", "sl"]],
                                    on="amplicon_id", how="left", validate="many_to_one")
    rounds = {label: r for r, label in enumerate(scene.round_labels)}
    index = {identity: i for i, identity in enumerate(scene.amplicon_ids)}
    channel, amplitude = [], []
    for row in table.itertuples():
        r = rounds[row.round_label]
        c = scene.codebook.color_to_channel[row.codeword[r]]
        channel.append(scene.channel_labels[c])
        amplitude.append(float(scene.realized[index[row.amplicon_id], c, r]))
    base = dict(zip(scene.codebook.table.gene_id, scene.codebook.table.get(
        "base_sequence", pd.Series([pd.NA] * scene.codebook.n_genes))))
    rendered = table.support_intersects & table.emitting
    reason = np.where(~table.emitting, "not_emitting",
                      np.where(~table.support_intersects, "outside_image",
                               np.where(~table.center_in_bounds, "center_outside_image", None)))
    return pd.DataFrame(dict(
        spot_namespace=fov, spot_id=table.amplicon_id.astype(str), round_label=table.round_label.astype(str),
        z=table.z, y=table.y, x=table.x, gene=table.gene_id.astype(str),
        barcode=[base[g] for g in table.gene_id], color_seq=table.codeword.astype(str),
        channel_label=channel, intensity=amplitude, sigma=table.sl, rendered=rendered.astype(bool),
        eligible=(rendered & table.center_in_bounds).astype(bool), eligibility_reason=reason,
        frame_id=table.frame_id.astype(str), units="voxel_index",
        perturbation_direction="reference_to_moving", molecular_truth_eligible=False))


def _fov_truth(scene, reference):
    """Historical v2 FOV record: reference positions, genes and round translations."""
    base = dict(zip(scene.codebook.table.gene_id, scene.codebook.table.get(
        "base_sequence", pd.Series([None] * scene.codebook.n_genes))))
    spots = [dict(id=row.amplicon_id, gene=row.gene_id, barcode=base[row.gene_id],
                  color_seq=row.codeword, position=[row.z, row.y, row.x], intensity=row.A)
             for row in scene.formed.itertuples()]
    transforms = {t["round_label"]: t for t in scene.provenance["transforms"].values()}
    shifts = {label: [float(v) for v in transforms[label]["translation_zyx"]]
              for label in scene.round_labels}
    record = dict(shifts=shifts, spots=spots, reference_round=reference)
    deformations = {label: dict(type=t["kind"], lipschitz_bound=t["lipschitz_bound"])
                    for label, t in transforms.items()
                    if np.any(t["vectors_zyx"]) or np.any(t.get("affine_zyx", 0))
                    or np.any(t.get("polynomial_zyx", 0))}
    if deformations:
        record["deformations"] = deformations
    return record


def generate_dataset(codebook: Codebook, config: FormedSceneConfig, *,
                     fov_ids: tuple[str, ...] = ("FOV_001",), preset: str | None = None,
                     on_round: Callable | None = None) -> SyntheticDataset:
    """Generate one formed scene per FOV ID and derive historical truth.

    Each FOV uses ``config`` with ``FOV_id`` set and its own stream namespace,
    scene key ``["<config.scene_key>", "<FOV ID>"]``; appending IDs never
    changes the draws of earlier FOVs. Rounds follow the codebook. The
    reference round is ``config.geometry.reference_round``, or the first round
    when that is None; it is held at identity, so ``formed`` positions are its
    positions. Draws of the other rounds do not depend on this choice. With
    ``on_round(fov_id, round_label, image, metadata)``, each round image is
    handed over as soon as it is generated and not retained, so at most one
    round is held in memory. ``preset`` is recorded in the historical truth.
    """
    if not isinstance(config, FormedSceneConfig):
        raise TypeError("config must be FormedSceneConfig")
    fov_ids = tuple(fov_ids)
    if not fov_ids or len(set(fov_ids)) != len(fov_ids):
        raise ValueError("fov_ids must be a nonempty tuple of unique labels")
    for fov in fov_ids:
        _label(fov)
    reference = config.geometry.reference_round or codebook.round_labels[0]
    if config.geometry.reference_round is None:
        # The reference round defines the truth frame, so it never moves.
        config = replace(config, geometry=replace(config.geometry, reference_round=reference))
    rounds, metadata, provenance, formed, truth, spots, records = {}, {}, {}, [], [], [], {}
    for fov in fov_ids:
        scene_config = replace(config, FOV_id=fov, scene_key=_json([config.scene_key, fov]))
        callback = None if on_round is None else (lambda label, image, meta, fov=fov:
                                                  on_round(fov, label, image, meta))
        scene = _generate(codebook, scene_config, None, callback)
        rounds[fov] = scene.rounds
        metadata[fov] = scene.round_metadata
        provenance[fov] = scene.provenance
        formed.append(scene.formed)
        truth.append(scene.round_truth)
        spots.append(_spot_truth(scene, fov))
        records[fov] = _fov_truth(scene, reference)
    historical = dict(version=HISTORICAL_TRUTH_VERSION, preset=preset, preset_version=PRESET_VERSION,
                      source="derived from formed.csv and the round transforms in generation.json",
                      seed=config.seed,
                      image_shape=list(config.shape_zyx), n_rounds=len(codebook.round_labels),
                      n_channels=len(codebook.channel_labels), n_genes=codebook.n_genes,
                      round_labels=list(codebook.round_labels), reference_round=reference, fovs=records)
    return SyntheticDataset(rounds if on_round is None else {fov: {} for fov in fov_ids}, metadata,
                            scene.codebook, pd.concat(formed, ignore_index=True),
                            pd.concat(truth, ignore_index=True), pd.concat(spots, ignore_index=True),
                            provenance, historical)


def generate_registration_pair(preset: str = "small", *, deformation: str = "shift",
                               seed: int | None = None, dtype: str = "uint16", noise: bool = True,
                               include_reference: bool = True,
                               on_round: Callable | None = None) -> SyntheticDataset:
    """Generate one reference/moving pair of a benchmark preset.

    Both rounds image the same formed scene; only the moving round (labelled
    by the deformation name) is moved, by a translation (``shift``) or one of
    DEFORMATION_PRESETS. The forward map is recorded in the scene transforms
    and summarized in ``historical_truth['pairs']``; use forward_displacement
    for the voxel field. The FOV key is the preset name. Signal is in ch00.
    ``seed`` None keeps the preset seed; ``noise=False`` disables both
    residuals. ``on_round(preset, round_label, image, metadata)`` streams
    rounds as in generate_dataset. Every pair of a preset shares the reference
    image; ``include_reference=False`` skips rendering it again (its truth
    rows are kept).
    """
    codebook, config = registration_scene_preset(preset, deformation, dtype=dtype)
    config = _with_seed(config, seed)
    if not noise:
        config = replace(config, noise=NoiseConfig())
    callback = None if on_round is None else (lambda label, image, meta:
                                              on_round(preset, label, image, meta))
    scene = _generate(codebook, config, None, callback, () if include_reference else ('reference',))
    moving = next(t for t in scene.provenance["transforms"].values() if t["round_label"] == deformation)
    if deformation == "shift":
        pair = dict(type="global_shift", shift_zyx=[float(v) for v in moving["translation_zyx"]])
    else:
        pair = dict(type="local_deformation", deformation_type=deformation, kind=moving["kind"],
                    lipschitz_bound=moving["lipschitz_bound"], field_file=f"field_{deformation}.npy")
    historical = dict(preset=preset, preset_version=PRESET_VERSION, shape=list(config.shape_zyx),
                      n_spots=len(scene.formed), seed=config.seed,
                      spot_positions=scene.formed[["z", "y", "x"]].to_numpy().tolist(),
                      pairs={deformation: pair})
    return SyntheticDataset({preset: scene.rounds} if on_round is None else {preset: {}},
                            {preset: scene.round_metadata}, scene.codebook, scene.formed,
                            scene.round_truth, _spot_truth(scene, preset), {preset: scene.provenance},
                            historical)


def forward_displacement(transform: dict, shape_zyx, *, z: slice | None = None) -> np.ndarray:
    """Evaluate a recorded forward map F(q) - q on reference voxels q.

    Returns a float32 array (Z×Y×X×3, ZYX voxel displacements) for the whole
    grid or for the Z planes selected by ``z``, evaluated block by block.
    ``transform`` is a record from a scene's ``provenance['transforms']``.
    """
    shape = tuple(int(n) for n in shape_zyx)
    planes = range(shape[0])[z if z is not None else slice(None)]
    if not planes or planes.step != 1:
        raise ValueError("z must select a nonempty contiguous range of planes")
    out = np.empty((len(planes), *shape[1:], 3), dtype=np.float32)
    for block in _blocks((len(planes), *shape[1:])):
        start = np.array([s.start for s in block], dtype=np.float64) + [planes.start, 0, 0]
        grid = np.moveaxis(np.indices(tuple(s.stop - s.start for s in block), dtype=np.float64), 0, -1) + start
        out[block] = _forward(grid, transform) - grid
    return out
