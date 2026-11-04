"""Plain files for amplicon scenes and datasets: images, CSV truth, JSON provenance.

Also the streaming writers of ``starfinder synthetic generate`` (its e2e dataset and registration pairs), its
manifest helpers and the green-magenta inspection image, which moved here from ``starfinder.benchmark`` (R7).
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from starfinder._records import file_sha256
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


# --- The synthetic command's writers (moved from starfinder.benchmark, R7) -------------

MANIFEST_SCHEMA_VERSION = 1


def _write(path, value):
    """Write strict JSON by exclusive creation: an existing file, even of an interrupted attempt, is never
    overwritten."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def _reference(root, path):
    """A file's path relative to ``root`` and its SHA-256."""
    return {'path': str(Path(path).relative_to(root)), 'sha256': file_sha256(path)}


def _generate_annotated_visualization(
    output_dir: Path,
    fov_id: str,
    fov_dir: Path,
    spots: list[dict],
    image_shape: tuple[int, int, int],
    n_channels: int,
) -> None:
    """Generate annotated max projection visualization with spot bounding boxes.

    ``spots`` are ground_truth.json spot records, with the historical ``gene``
    and ``color_seq`` keys.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import tifffile

    round1_dir = fov_dir / "round1"
    max_proj = None

    for ch in range(n_channels):
        img = tifffile.imread(round1_dir / f"ch{ch:02d}.tif")
        ch_max = img.max(axis=0)
        if max_proj is None:
            max_proj = ch_max.astype(np.float32)
        else:
            max_proj = np.maximum(max_proj, ch_max)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(max_proj, cmap="gray", vmin=0, vmax=max_proj.max())
    ax.set_title(f"{fov_id} - Round 1 Max Projection (all channels)", fontsize=14)

    unique_genes = sorted(set(s["gene"] for s in spots))
    cmap = plt.cm.get_cmap("tab20", max(len(unique_genes), 8))
    gene_colors = {g: cmap(i % cmap.N) for i, g in enumerate(unique_genes)}

    box_size = 12
    annotate = len(spots) <= 50
    for spot in spots:
        _, y, x = spot["position"]
        gene = spot["gene"]
        color_sequence = spot["color_seq"]
        color = gene_colors[gene]

        rect = patches.Rectangle(
            (x - box_size // 2, y - box_size // 2),
            box_size,
            box_size,
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        if annotate:
            label = f"{gene}\n{color_sequence}"
            ax.annotate(
                label,
                (x, y - box_size // 2 - 2),
                fontsize=6,
                color=color,
                ha="center",
                va="bottom",
                weight="bold",
            )

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_xlim(0, image_shape[2])
    ax.set_ylim(image_shape[1], 0)

    plt.tight_layout()
    output_path = output_dir / f"ground_truth_annotation_{fov_id}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_dataset(preset, output_dir, *, seed=None, dtype='uint16', noise=True, fov_ids=None,
                   annotations=False):
    """Generate an e2e preset round by round into the MATLAB-compatible layout.

    Writes ``<FOV>/<round>/<channel>.tif`` (ZYX) as each round is generated,
    then codebook.csv (gene,barcode), ground_truth.json (historical v2 keys
    derived from formed and the round transforms), scene_truth.csv,
    formed.csv, round_truth.csv and generation.json. Returns the
    SyntheticDataset without images.
    """
    from dataclasses import replace
    from starfinder.synthetic import (BENCHMARK_PRESETS, NoiseConfig, benchmark_scene_preset,
                                      generate_amplicon_dataset)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    codebook, config = benchmark_scene_preset(preset, dtype=dtype)
    config = replace(config, seed=config.seed if seed is None else seed,
                     **({} if noise else dict(noise=NoiseConfig())))
    if fov_ids is None:
        fov_ids = tuple(f'FOV_{i + 1:03d}' for i in range(BENCHMARK_PRESETS[preset]['fovs']))

    def write(fov, label, image, metadata):
        _write_channels(output_dir / fov / label, image, codebook.channel_labels)

    result = generate_amplicon_dataset(codebook, config, fov_ids=fov_ids, preset=preset, on_round=write)
    with (output_dir / 'codebook.csv').open('w') as stream:
        stream.write('gene,barcode\n')
        for gene, barcode in zip(codebook.table.gene_id, codebook.table.base_sequence):
            stream.write(f'{gene},{barcode}\n')
    _write_truth(output_dir, result, _amplicon_generation(preset=preset, seed=config.seed, dtype=config.dtype,
                                                          noise=noise, fov_ids=fov_ids))
    if annotations:
        for fov, data in result.historical_truth['fovs'].items():
            _generate_annotated_visualization(output_dir, fov, output_dir / fov, data['spots'],
                                              config.shape_zyx, len(codebook.channel_labels))
    return result


def _write_field(path, transform, shape):
    """Write a float32 Z×Y×X×3 forward displacement .npy one Z plane at a time."""
    from starfinder.synthetic import forward_displacement
    with open(path, 'wb') as stream:
        np.lib.format.write_array_header_1_0(stream, dict(descr='<f4', fortran_order=False,
                                                          shape=(*shape, 3)))
        for z in range(shape[0]):
            stream.write(forward_displacement(transform, shape, z=slice(z, z + 1)).astype('<f4').tobytes())


def _write_registration_pairs(preset, output_dir, *, seed=None, dtype='uint16', noise=True,
                              deformations=None, inspections=False):
    """Generate every registration pair of a preset into ``synthetic/<preset>/``.

    Files: ref.tif, mov_shift.tif, mov_deform_<name>.tif (ZYX, ch00),
    field_<name>.npy (float32 forward displacement F(q)-q on the reference
    grid), formed.csv, round_truth.csv, scene_truth.csv, ground_truth.json,
    generation.json, and synthetic/summary.json. All pairs share one scene, so
    the reference image and amplicons are written once.
    """
    import tifffile
    from starfinder.synthetic import DEFORMATION_PRESETS, generate_registration_pair
    directory = Path(output_dir) / 'synthetic' / preset
    directory.mkdir(parents=True, exist_ok=True)
    deformations = ('shift', *DEFORMATION_PRESETS) if deformations is None else tuple(deformations)
    images, results = {}, []
    for i, deformation in enumerate(deformations):
        def write(_, label, image, metadata):
            name = 'ref' if label == 'reference' else ('mov_shift' if label == 'shift' else f'mov_deform_{label}')
            volume = np.ascontiguousarray(image[..., 0])
            tifffile.imwrite(directory / f'{name}.tif', volume, imagej=True, metadata={'axes': 'ZYX'})
            if inspections:
                images[label] = volume
        result = generate_registration_pair(preset, deformation=deformation, seed=seed, dtype=dtype, noise=noise,
                                            include_reference=i == 0, on_round=write)
        transform = next(t for t in result.provenance[preset]['transforms'].values()
                         if t['round_label'] == deformation)
        shape = tuple(result.historical_truth['shape'])
        if deformation != 'shift':
            _write_field(directory / f'field_{deformation}.npy', transform, shape)
        if inspections:
            suffix = 'shift' if deformation == 'shift' else f'deform_{deformation}'
            generate_inspection_image(images['reference'], images.pop(deformation),
                                      {'preset': preset, **result.historical_truth['pairs'][deformation]},
                                      directory / f'inspection_{suffix}.png')
        results.append(result)
    first = results[0]
    reference = lambda table, column: table[column].astype(str) != 'reference'  # noqa: E731
    merged = type(first)(
        {}, {preset: {k: v for r in results for k, v in r.metadata[preset].items()}}, first.codebook,
        first.formed,
        pd.concat([first.round_truth] + [r.round_truth[reference(r.round_truth, 'round_label')]
                                         for r in results[1:]], ignore_index=True),
        pd.concat([first.spot_truth] + [r.spot_truth[reference(r.spot_truth, 'round_label')]
                                        for r in results[1:]], ignore_index=True),
        {d: r.provenance[preset] for d, r in zip(deformations, results)},
        dict(first.historical_truth, pairs={k: v for r in results for k, v in r.historical_truth['pairs'].items()}))
    _write_truth(directory, merged, dict(generator='starfinder.synthetic.generate_registration_pair',
        generator_version=_GENERATOR_VERSION, mode='registration', preset=preset,
        preset_version=PRESET_VERSION, seed=first.historical_truth['seed'], dtype=dtype, noise=noise,
        deformations=list(deformations)))
    summary = {'seed': first.historical_truth['seed'], 'presets': {preset: {
        'shape': first.historical_truth['shape'], 'n_spots': first.historical_truth['n_spots'],
        'n_pairs': len(deformations)}}}
    _json_dump(Path(output_dir) / 'synthetic' / 'summary.json', summary)
    return summary


def generate_inspection_image(
    ref: np.ndarray,
    mov: np.ndarray,
    ground_truth: dict,
    output_path: Path,
) -> None:
    """Generate a green-magenta composite inspection image.

    Parameters
    ----------
    ref : np.ndarray
        Reference volume (Z, Y, X).
    mov : np.ndarray
        Moving volume (Z, Y, X).
    ground_truth : dict
        Ground truth metadata.
    output_path : Path
        Path to save the inspection image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Maximum intensity projections
    ref_mip = np.max(ref, axis=0).astype(np.float32)
    mov_mip = np.max(mov, axis=0).astype(np.float32)

    # Normalize to [0, 1]
    ref_norm = ref_mip / ref_mip.max() if ref_mip.max() > 0 else ref_mip
    mov_norm = mov_mip / mov_mip.max() if mov_mip.max() > 0 else mov_mip

    # Green-magenta composite (green=ref, magenta=mov)
    composite = np.stack([mov_norm, ref_norm, mov_norm], axis=-1)

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(ref_mip, cmap="gray")
    axes[0].set_title(f"Reference MIP\nShape: {ref.shape}")
    axes[0].axis("off")

    axes[1].imshow(mov_mip, cmap="gray")
    axes[1].set_title(f"Moving MIP\nShape: {mov.shape}")
    axes[1].axis("off")

    axes[2].imshow(composite)
    axes[2].set_title("Overlay (G=ref, M=mov)\nWhite=aligned, Color=misaligned")
    axes[2].axis("off")

    # Ground truth info
    info_lines = [f"Dataset: {ground_truth.get('preset', 'N/A')}"]
    if "shift_zyx" in ground_truth:
        info_lines.append(f"Shift (Z,Y,X): {ground_truth['shift_zyx']}")
    if "deformation_type" in ground_truth:
        info_lines.append(f"Deformation: {ground_truth['deformation_type']}")
        info_lines.append(f"Max disp: {ground_truth.get('max_displacement', 'N/A')} px")

    info_text = "\n".join(info_lines)
    axes[3].text(
        0.1, 0.5, info_text,
        fontsize=12, family="monospace",
        verticalalignment="center",
        transform=axes[3].transAxes,
    )
    axes[3].axis("off")
    axes[3].set_title("Ground Truth")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
