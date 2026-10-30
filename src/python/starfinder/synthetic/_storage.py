"""Plain files for amplicon scenes and datasets: images, CSV truth, JSON provenance."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from starfinder.io import save_volume

from ._datasets import _HISTORICAL_NAMES, SyntheticDataset
from ._formed import _GENERATOR_VERSION, AmpliconScene
from ._presets import PRESET_VERSION


def save_amplicon_scene(scene: AmpliconScene, directory: Path | str) -> Path:
    """Write one amplicon scene to a new or empty directory and return its path.

    Parameters
    ----------
    scene : AmpliconScene
        Scene containing round images, truth tables, signal tensors and provenance to
        save.
    directory : pathlib.Path or str
        Destination directory for the scene files.

    Returns
    -------
    pathlib.Path
        Path of the directory containing the saved scene.

    Notes
    -----
    Files: ``images/<round_label>.ome.tif`` (ZYXC OME-TIFF in the scene dtype,
    written by ``starfinder.io.save_volume`` with that round's ``round_metadata``),
    ``formed.csv``, ``round_truth.csv``, ``signals.csv`` (one row per amplicon,
    channel and round with intended, pre_mix and realized amplitudes) and
    ``provenance.json``. Round labels must be plain file names. CSV floats use
    round-trip precision; missing ``first_loss_round`` values are empty cells.
    An existing nonempty directory raises FileExistsError; nothing is overwritten.
    """
    if not isinstance(scene, AmpliconScene):
        raise TypeError("scene must be AmpliconScene")
    for label in scene.round_labels:
        if label in (".", "..") or Path(label).name != label or "\\" in label:
            raise ValueError(f"round label is not a plain file name: {label!r}")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"fixture directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    metadata = scene.round_metadata
    for label, image in scene.rounds.items():
        save_volume(image, directory / "images" / f"{label}.ome.tif", metadata=metadata[label])
    scene.formed.to_csv(directory / "formed.csv", index=False)
    scene.round_truth.to_csv(directory / "round_truth.csv", index=False)
    rows = [dict(amplicon_id=identity, channel_label=channel, round_label=label,
                 intended=scene.intended[i, c, r], pre_mix=scene.pre_mix[i, c, r],
                 realized=scene.realized[i, c, r])
            for i, identity in enumerate(scene.amplicon_ids)
            for c, channel in enumerate(scene.channel_labels)
            for r, label in enumerate(scene.round_labels)]
    columns = ["amplicon_id", "channel_label", "round_label", "intended", "pre_mix", "realized"]
    pd.DataFrame(rows, columns=columns).to_csv(directory / "signals.csv", index=False)
    (directory / "provenance.json").write_text(
        json.dumps(scene.provenance, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    return directory


def _json_dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _compact_streams(provenance):
    """Scene provenance with the per-stream descriptor list replaced by counts.

    Descriptors follow from the documented key scheme and are about 1 kB each;
    a tissue-sized FOV draws ~10^5 streams, so files keep counts per component.
    """
    scheme = provenance.get('stream_scheme')
    if not isinstance(scheme, dict) or 'streams' not in scheme:
        return provenance
    streams = scheme['streams']
    components = {}
    for descriptor in streams:
        components[descriptor[4]] = components.get(descriptor[4], 0) + 1
    scheme = {k: v for k, v in scheme.items() if k != 'streams'}
    return dict(provenance, stream_scheme=dict(scheme, stream_count=len(streams),
                                               streams_per_component=dict(sorted(components.items()))))


def _write_truth(directory, result, generation):
    """Truth tables shared by every dataset writer; generation.json holds requested configs/provenance.

    formed.csv and round_truth.csv keep the Python names. scene_truth.csv and
    ground_truth.json are MATLAB-compatible and keep the historical ``gene``
    and ``color_seq`` names (the synthetic ``_HISTORICAL_NAMES`` mapping).
    """
    result.formed.to_csv(directory / 'formed.csv', index=False)
    result.round_truth.to_csv(directory / 'round_truth.csv', index=False)
    result.spot_truth.rename(columns=_HISTORICAL_NAMES).to_csv(directory / 'scene_truth.csv', index=False)
    _json_dump(directory / 'ground_truth.json', result.historical_truth)
    provenance = {key: _compact_streams(value) for key, value in result.provenance.items()}
    _json_dump(directory / 'generation.json', dict(generation, provenance=provenance,
        metadata={f: {r: asdict(m) for r, m in rounds.items()} for f, rounds in result.metadata.items()},
        molecular_truth=None))


def _amplicon_generation(*, preset, seed, dtype, noise, fov_ids):
    """The generation.json fields of a dataset from generate_amplicon_dataset (mode e2e)."""
    return dict(generator='starfinder.synthetic.generate_amplicon_dataset', generator_version=_GENERATOR_VERSION,
                mode='e2e', preset=preset, preset_version=PRESET_VERSION, seed=seed, dtype=dtype, noise=noise,
                fov_ids=list(fov_ids))


def _write_channels(directory, image, channel_labels):
    """Write a ZYXC round image as one ZYX ImageJ TIFF per channel, ``<directory>/<channel>.tif``."""
    import tifffile
    directory.mkdir(parents=True, exist_ok=True)
    for c, channel in enumerate(channel_labels):
        tifffile.imwrite(directory / f'{channel}.tif', np.ascontiguousarray(image[..., c]),
                         imagej=True, metadata={'axes': 'ZYX'})


def save_amplicon_dataset(dataset: SyntheticDataset, directory: Path | str) -> Path:
    """Write an in-memory amplicon dataset as a raw dataset that ``Dataset`` and ``FOV`` read; return its path.

    Parameters
    ----------
    dataset : SyntheticDataset
        In-memory amplicon dataset from generate_amplicon_dataset, including its round
        images.
    directory : pathlib.Path or str
        Destination directory for the raw dataset and truth files.

    Returns
    -------
    pathlib.Path
        Path of the directory containing the saved dataset.

    Notes
    -----
    ``dataset`` is a ``SyntheticDataset`` from ``generate_amplicon_dataset`` that holds its round
    images. Layout: ``<round>/<fov_id>/<channel>.tif`` (one ZYX ImageJ TIFF per codebook channel),
    ``codebook.csv`` (``entry_id``, ``gene_id``, ``color_sequence``; the entry ID is the base
    sequence when the codebook has one, as a ``gene,barcode`` codebook names it), and the truth files
    ``formed.csv``, ``round_truth.csv``, ``scene_truth.csv``, ``ground_truth.json`` and
    ``generation.json``. The images and truth files are those the streaming writer of
    ``starfinder synthetic generate --mode e2e`` writes for the same dataset; in generation.json,
    ``preset`` and ``seed`` come from the historical truth, ``dtype`` from the requested config and
    ``noise`` records whether its noise config enables a residual. No ``dataset.json`` is written.
    A dataset generated with ``on_round``, so that a FOV holds no round images, raises ValueError,
    as does one not from ``generate_amplicon_dataset``; an existing nonempty directory raises
    FileExistsError, and nothing is overwritten.
    """
    if not isinstance(dataset, SyntheticDataset):
        raise TypeError("dataset must be SyntheticDataset")
    truth = dataset.historical_truth
    if not isinstance(truth.get("fovs"), dict):
        raise ValueError("dataset is not from generate_amplicon_dataset (its historical truth has no FOVs)")
    fov_ids = list(truth["fovs"])
    round_labels = list(dataset.codebook.round_labels)
    if any(set(dataset.rounds.get(fov, {})) != set(round_labels) for fov in fov_ids):
        raise ValueError("the dataset holds no round images (generated with on_round); write each round in the "
                         "callback")
    for name in (*fov_ids, *round_labels):
        if name in (".", "..") or Path(name).name != name or "\\" in name:
            raise ValueError(f"FOV ID or round is not a plain file name: {name!r}")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"dataset directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    for fov in fov_ids:
        for label in round_labels:
            _write_channels(directory / label / fov, dataset.rounds[fov][label], dataset.channel_labels)
    table = dataset.codebook.table
    # The streaming writer's gene,barcode codebook names each entry by its barcode as written, and
    # Dataset.load_codebook reads it so; the same entry IDs keep decoded tables equal between the two.
    entries = table["base_sequence"] if "base_sequence" in table else table["entry_id"]
    pd.DataFrame(dict(entry_id=entries, gene_id=table["gene_id"], color_sequence=table["color_sequence"])).to_csv(
        directory / "codebook.csv", index=False)
    requested = dataset.provenance[fov_ids[0]]["requested_config"]
    noise = requested["noise"]
    enabled = bool(noise["dependent_enabled"] or noise["independent_enabled"] or noise["correlated_enabled"])
    _write_truth(directory, dataset, _amplicon_generation(preset=truth["preset"], seed=truth["seed"],
                                                          dtype=requested["dtype"], noise=enabled, fov_ids=fov_ids))
    return directory
