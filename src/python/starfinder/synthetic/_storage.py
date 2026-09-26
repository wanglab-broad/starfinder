"""Plain fixture files for formed scenes: TIFF images, CSV truth, JSON provenance."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from starfinder.io import save_volume

from ._formed import FormedScene


def save_formed_scene(scene: FormedScene, directory: Path | str) -> Path:
    """Write one formed scene to a new or empty directory and return its path.

    Files: ``images/<round_label>.tif`` (ZYXC in the scene dtype, written by
    ``starfinder.io.save_volume`` with that round's ``round_metadata``),
    ``formed.csv``, ``round_truth.csv``, ``signals.csv`` (one row per amplicon,
    channel and round with intended, pre_mix and realized amplitudes) and
    ``provenance.json``. Round labels must be plain file names. CSV floats use
    round-trip precision; missing ``first_loss_round`` values are empty cells.
    An existing nonempty directory raises FileExistsError; nothing is overwritten.
    """
    if not isinstance(scene, FormedScene):
        raise TypeError("scene must be FormedScene")
    for label in scene.round_labels:
        if label in (".", "..") or Path(label).name != label or "\\" in label:
            raise ValueError(f"round label is not a plain file name: {label!r}")
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"fixture directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    metadata = scene.round_metadata
    for label, image in scene.rounds.items():
        save_volume(image, directory / "images" / f"{label}.tif", metadata=metadata[label])
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
