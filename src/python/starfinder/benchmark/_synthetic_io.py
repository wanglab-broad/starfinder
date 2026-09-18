"""Synthetic persistence/reporting adapters; generation performs no I/O."""
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

def _write_dataset(result, output_dir, *, annotations=True):
    """Persist generated arrays and legacy v2 layout for workflow consumers."""
    from dataclasses import asdict
    import tifffile
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fov, rounds in result.rounds.items():
        for label, image in rounds.items():
            directory = output_dir / fov / label
            directory.mkdir(parents=True, exist_ok=True)
            for c, channel in enumerate(result.channel_labels):
                tifffile.imwrite(directory / f'{channel}.tif', image[..., c],
                                 imagej=True, metadata={'axes': 'ZYX'})
        if fov in result.perturbations and 'field' in result.perturbations[fov]:
            np.save(output_dir / fov / 'deformation_field.npy', result.perturbations[fov]['field'])
    with (output_dir / 'codebook.csv').open('w') as stream:
        stream.write('gene,barcode\n')
        for gene, barcode in result.codebook:
            stream.write(f'{gene},{barcode}\n')
    (output_dir / 'ground_truth.json').write_text(json.dumps(result.historical_truth, indent=2))
    result.spot_truth.to_csv(output_dir / 'scene_truth.csv', index=False)
    (output_dir / 'generation.json').write_text(json.dumps({
        'config': asdict(result.config), 'provenance': result.provenance,
        'metadata': {f: {r: asdict(m) for r, m in rounds.items()} for f, rounds in result.metadata.items()},
        'molecular_truth': None,
    }, indent=2))
    if annotations:
        for fov, data in result.historical_truth['fovs'].items():
            _generate_annotated_visualization(output_dir, fov, output_dir / fov,
                data['spots'], result.config.shape_zyx, result.config.n_channels)


def _write_registration_pairs(results, output_dir, *, inspections=True):
    """Persist already generated pairs; never rerun generation."""
    import tifffile
    from .data import generate_inspection_image
    summary = {'presets': {}}
    for preset, result in results.items():
        directory = Path(output_dir) / 'synthetic' / preset
        directory.mkdir(parents=True, exist_ok=True)
        rounds = result.rounds[preset]
        for label, image in rounds.items():
            filename = 'ref' if label == 'reference' else ('mov_shift' if label == 'shift' else f'mov_deform_{label}')
            tifffile.imwrite(directory / f'{filename}.tif', image, imagej=True, metadata={'axes': 'ZYX'})
            if label in result.perturbations and 'field' in result.perturbations[label]:
                np.save(directory / f'field_{label}.npy', result.perturbations[label]['field'])
            if inspections and label != 'reference':
                info = {'preset': preset, **result.historical_truth['pairs'][label]}
                suffix = 'shift' if label == 'shift' else f'deform_{label}'
                generate_inspection_image(rounds['reference'], image, info, directory / f'inspection_{suffix}.png')
        (directory / 'ground_truth.json').write_text(json.dumps(result.historical_truth, indent=2))
        result.spot_truth.to_csv(directory / 'scene_truth.csv', index=False)
        (directory / 'generation.json').write_text(json.dumps({'config': result.config, 'provenance': result.provenance}, indent=2))
        summary['seed'] = result.provenance['seed']
        summary['presets'][preset] = {'shape': list(result.config['shape_zyx']),
            'n_spots': result.config['n_spots'], 'n_pairs': len(rounds) - 1}
    directory = Path(output_dir) / 'synthetic'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
