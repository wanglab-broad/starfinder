"""Synthetic persistence/reporting adapters: stream generated rounds to files."""
from pathlib import Path
import json
import numpy as np

def _generate_annotated_visualization(
    output_dir: Path,
    fov_id: str,
    fov_dir: Path,
    spots: list[dict],
    image_shape: tuple[int, int, int],
    n_channels: int,
) -> None:
    """Generate annotated max projection visualization with spot bounding boxes."""
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
        color_seq = spot["color_seq"]
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
            label = f"{gene}\n{color_seq}"
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
    """Truth tables shared by both modes; generation.json holds requested configs/provenance."""
    from dataclasses import asdict
    result.formed.to_csv(directory / 'formed.csv', index=False)
    result.round_truth.to_csv(directory / 'round_truth.csv', index=False)
    result.spot_truth.to_csv(directory / 'scene_truth.csv', index=False)
    _json_dump(directory / 'ground_truth.json', result.historical_truth)
    provenance = {key: _compact_streams(value) for key, value in result.provenance.items()}
    _json_dump(directory / 'generation.json', dict(generation, provenance=provenance,
        metadata={f: {r: asdict(m) for r, m in rounds.items()} for f, rounds in result.metadata.items()},
        molecular_truth=None))


def _write_dataset(preset, output_dir, *, seed=None, dtype='uint16', noise=True, fov_ids=None,
                   annotations=False):
    """Generate an e2e preset round by round into the MATLAB-compatible layout.

    Writes ``<FOV>/<round>/<channel>.tif`` (ZYX) as each round is generated,
    then codebook.csv (gene,barcode), ground_truth.json (historical v2 keys
    derived from round_truth), scene_truth.csv, formed.csv, round_truth.csv and
    generation.json. Returns the SyntheticDataset without images.
    """
    from dataclasses import replace
    import tifffile
    from starfinder.synthetic import (BENCHMARK_PRESETS, NoiseConfig, PRESET_VERSION,
                                      benchmark_scene_preset, generate_dataset)
    from starfinder.synthetic._formed import _GENERATOR_VERSION
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    codebook, config = benchmark_scene_preset(preset, dtype=dtype)
    config = replace(config, seed=config.seed if seed is None else seed,
                     **({} if noise else dict(noise=NoiseConfig())))
    if fov_ids is None:
        fov_ids = tuple(f'FOV_{i + 1:03d}' for i in range(BENCHMARK_PRESETS[preset]['fovs']))

    def write(fov, label, image, metadata):
        directory = output_dir / fov / label
        directory.mkdir(parents=True, exist_ok=True)
        for c, channel in enumerate(codebook.channel_labels):
            tifffile.imwrite(directory / f'{channel}.tif', np.ascontiguousarray(image[..., c]),
                             imagej=True, metadata={'axes': 'ZYX'})

    result = generate_dataset(codebook, config, fov_ids=fov_ids, preset=preset, on_round=write)
    with (output_dir / 'codebook.csv').open('w') as stream:
        stream.write('gene,barcode\n')
        for gene, barcode in zip(codebook.table.gene_id, codebook.table.base_sequence):
            stream.write(f'{gene},{barcode}\n')
    _write_truth(output_dir, result, dict(generator='starfinder.synthetic.generate_dataset',
        generator_version=_GENERATOR_VERSION, mode='e2e', preset=preset, preset_version=PRESET_VERSION,
        seed=config.seed, dtype=config.dtype, noise=noise, fov_ids=list(fov_ids)))
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
    import pandas as pd
    from starfinder.synthetic import DEFORMATION_PRESETS, PRESET_VERSION, generate_registration_pair
    from starfinder.synthetic._formed import _GENERATOR_VERSION
    from ._reporting import generate_inspection_image
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
