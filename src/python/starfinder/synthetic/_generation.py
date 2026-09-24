"""Pure scene generation; historical numerical choices are intentionally retained."""
from copy import deepcopy
import numpy as np
from starfinder.barcode import EncodingConfig
from starfinder.image import ImageMetadata
from ._config import SyntheticConfig
from ._presets import (_TEST_CODEBOOK, SIZE_PRESETS, SPOT_COUNTS, SHIFT_RANGES,
    DEFORMATION_CONFIGS, _scale_deformation_config, get_preset_config)
from ._fields import generate_displacement_field
from ._perturbations import _apply_shift_to_spots, _apply_deformation_to_spots
from ._rendering import render_spots
from ._truth import SyntheticDataset, LIMITATIONS, _scene_table, _truth_rows, _truth_table

def generate_dataset(config: SyntheticConfig | None = None, *, preset: str = "small") -> SyntheticDataset:
    """Generate ordered FOV/round images and explicit, unqualified scene truth.

    No files are written. Config overrides preset; ``historical_truth`` retains
    the old v2 records. All per-round dropout rows remain in ``spot_truth``.
    """
    if config is None:
        config = get_preset_config(preset)

    config = deepcopy(config)
    rounds, metadata, truth_rows, perturbations = {}, {}, [], {}

    rng = np.random.default_rng(config.seed)
    shape = (config.n_z, config.height, config.width)

    # Resolve codebook
    codebook = config.codebook if config.codebook is not None else _TEST_CODEBOOK

    # Resolve deformation field if configured
    deformation_field = None
    if config.deformation and config.deformation in DEFORMATION_CONFIGS:
        deform_config = DEFORMATION_CONFIGS[config.deformation]
        scaled = _scale_deformation_config(deform_config, shape)
        deformation_field = generate_displacement_field(
            shape=shape,
            deform_type=scaled["type"],
            max_displacement=scaled["max_displacement"],
            seed=config.seed + 99999,
            **{k: v for k, v in scaled.items() if k not in ["type", "max_displacement"]},
        )

    # Prepare ground truth structure
    ground_truth = {
        "version": "2.0",
        "preset": preset,
        "seed": config.seed,
        "image_shape": list(shape),
        "n_rounds": config.n_rounds,
        "n_channels": config.n_channels,
        "n_genes": len(codebook),
        "fovs": {},
    }

    # Generate each FOV
    for fov_idx in range(config.n_fovs):
        fov_id = f"FOV_{fov_idx + 1:03d}"
        rounds[fov_id], metadata[fov_id] = {}, {}

        # Generate random shifts for each round (round1 is reference)
        shifts = {"round1": [0, 0, 0]}
        for r in range(2, config.n_rounds + 1):
            shifts[f"round{r}"] = [
                int(rng.integers(-config.max_shift_z, config.max_shift_z + 1)),
                int(rng.integers(-config.max_shift_xy, config.max_shift_xy + 1)),
                int(rng.integers(-config.max_shift_xy, config.max_shift_xy + 1)),
            ]

        # Generate random spot positions and gene assignments
        spots_info = []
        margin_z = min(1, config.n_z // 4)
        margin_xy = min(10, config.height // 10)

        for spot_idx in range(config.n_spots_per_fov):
            gene, barcode = codebook[rng.integers(0, len(codebook))]
            color_seq = EncodingConfig(reverse_bases=True).encode(barcode)

            z = int(rng.integers(margin_z, max(margin_z + 1, config.n_z - margin_z)))
            y = int(rng.integers(margin_xy, config.height - margin_xy))
            x = int(rng.integers(margin_xy, config.width - margin_xy))
            intensity = int(rng.integers(config.spot_intensity[0], config.spot_intensity[1] + 1))

            spots_info.append({
                "id": spot_idx,
                "gene": gene,
                "barcode": barcode,
                "color_seq": color_seq,
                "position": [z, y, x],
                "intensity": intensity,
            })

        # Generate images for each round and channel using coordinate-first rendering
        for round_idx in range(1, config.n_rounds + 1):
            round_id = f"round{round_idx}"
            channels = []
            metadata[fov_id][round_id] = ImageMetadata(f"{fov_id}/{round_id}")

            shift = tuple(shifts[round_id])

            for ch in range(config.n_channels):
                # Collect spots for this channel with per-round jitter
                channel_spots: list[tuple] = []
                for spot in spots_info:
                    color = spot["color_seq"][round_idx - 1]
                    spot_channel = int(color) - 1
                    if spot_channel == ch:
                        z, y, x = spot["position"]

                        # Per-round intensity/sigma jitter (deterministic per spot+round)
                        jitter_rng = np.random.default_rng(
                            config.seed + spot["id"] * 100 + round_idx
                        )
                        jittered_intensity = max(1, int(
                            spot["intensity"] * (1 + jitter_rng.normal(0, 0.1))
                        ))
                        jittered_sigma = max(0.5, float(
                            config.spot_sigma * (1 + jitter_rng.normal(0, 0.05))
                        ))

                        channel_spots.append((z, y, x, jittered_intensity, jittered_sigma))

                # Truth is tracked separately, without changing rendering order or RNG calls.
                records = _truth_rows(channel_spots, namespace=fov_id,
                    round_label=round_id, shape=shape, shift=shift,
                    field=deformation_field if round_idx > 1 else None)
                selected = [s for s in spots_info if int(s['color_seq'][round_idx - 1]) - 1 == ch]
                for record, spot in zip(records, selected):
                    record.update(spot_id=spot['id'], gene=spot['gene'], barcode=spot['barcode'],
                                  color_seq=spot['color_seq'], channel_label=f'ch{ch:02d}')
                truth_rows.extend(records)
                # Apply coordinate transforms (shift, then deformation)
                if round_idx > 1:
                    channel_spots = _apply_shift_to_spots(channel_spots, shift, shape)
                    if config.deformation and deformation_field is not None:
                        channel_spots = _apply_deformation_to_spots(
                            channel_spots, deformation_field, shape
                        )

                # Render clean Gaussians at transformed positions
                image = render_spots(
                    shape=shape,
                    spots=_scene_table(channel_spots),
                    background=config.background_mean,
                    noise_std=config.noise_std,
                    seed=config.seed + fov_idx * 1000 + round_idx * 100 + ch,
                    add_noise=config.add_noise,
                    dtype=config.dtype,
                )

                channels.append(image)
            # Historical configurations may omit a codebook color channel.
            # Preserve the absent image signal and retain the excluded identity.
            for spot in spots_info:
                ch = int(spot['color_seq'][round_idx - 1]) - 1
                if ch >= config.n_channels:
                    record = _truth_rows([(*spot['position'], np.nan, np.nan)],
                        namespace=fov_id, round_label=round_id, shape=shape)[0]
                    record.update(spot_id=spot['id'], gene=spot['gene'], barcode=spot['barcode'],
                        color_seq=spot['color_seq'], channel_label=f'ch{ch:02d}',
                        rendered=False, eligible=False, eligibility_reason='channel_not_rendered')
                    for column in ('z', 'y', 'x', 'continuous_z', 'continuous_y', 'continuous_x'):
                        record[column] = np.nan  # no rendering/transformation was performed
                    truth_rows.append(record)
            rounds[fov_id][round_id] = np.stack(channels, axis=-1)

        # Build FOV ground truth
        fov_gt: dict = {
            "shifts": shifts,
            "spots": spots_info,
        }
        perturbations[fov_id] = {"shifts": shifts, "direction": "reference_to_moving",
            "units": "voxel_index", "reference_frame": f"{fov_id}/round1"}
        if config.deformation and deformation_field is not None:
            perturbations[fov_id].update(field=deformation_field,
                representation="forward_displacement_at_shifted_integer_center")
            fov_gt["deformations"] = {
                f"round{r}": {
                    "type": config.deformation,
                    "field_file": f"{fov_id}/deformation_field.npy",
                    "max_displacement": float(np.max(np.linalg.norm(
                        deformation_field, axis=-1
                    ))),
                }
                for r in range(2, config.n_rounds + 1)
            }

        ground_truth["fovs"][fov_id] = fov_gt

    return SyntheticDataset(rounds, metadata, tuple(f'ch{c:02d}' for c in range(config.n_channels)),
        _truth_table(truth_rows), None, config,
        {'seed': config.seed, 'limitations': LIMITATIONS,
         'encoding': {'reverse_bases': True}, 'background_std_used': False},
        perturbations, ground_truth, list(codebook))


def generate_registration_pairs(presets: list[str], *, seed: int = 42, add_noise: bool = True) -> dict[str, SyntheticDataset]:
    """Generate reference/moving scene collections without persistence or reporting.

    Select presets explicitly. Forward displacement fields describe scene
    movement, not inverse registration pull fields. Process hash seeds remain
    unqualified; resolved seeds are recorded in provenance.
    """
    results = {}

    for preset in presets:
        if preset not in SIZE_PRESETS:
            raise ValueError(f"Unknown preset: {preset}")

        shape = SIZE_PRESETS[preset]
        n_spots = SPOT_COUNTS.get(preset, 100)
        shift_range = SHIFT_RANGES.get(preset, {"z": (-5, 5), "yx": (-20, 20)})

        rounds, metadata, perturbations = {}, {}, {}
        truth_rows = []
        effective_seeds = {"spots": seed, "reference": seed, "shift_image": seed + 1, "deformed_image": seed + 2}

        # Generate spot positions (reference)
        spot_sigma = 1.5
        rng_spots = np.random.default_rng(seed)
        z_size, y_size, x_size = shape
        margin_z = 2
        margin_xy = 5

        spot_positions: list[tuple] = []
        for _ in range(n_spots):
            z = int(rng_spots.integers(margin_z, max(margin_z + 1, z_size - margin_z)))
            y = int(rng_spots.integers(margin_xy, max(margin_xy + 1, y_size - margin_xy)))
            x = int(rng_spots.integers(margin_xy, max(margin_xy + 1, x_size - margin_xy)))
            intensity = int(200 + rng_spots.integers(-20, 21))
            spot_positions.append((z, y, x, intensity, spot_sigma))

        # Render and save reference
        ref = render_spots(
            shape=shape,
            spots=_scene_table(spot_positions),
            background=20,
            noise_std=5,
            seed=seed,
            add_noise=add_noise,
        )
        rounds['reference'] = ref
        truth_rows.extend(_truth_rows(spot_positions, namespace=preset, round_label='reference', shape=shape))

        ground_truth: dict = {
            "preset": preset,
            "shape": list(shape),
            "n_spots": n_spots,
            "spot_positions": [(z, y, x) for z, y, x, _, _ in spot_positions],
            "seed": seed,
            "pairs": {},
        }

        # Generate shifted moving image using coordinate transform
        preset_seed = seed + hash(preset) % 10000
        rng_shift = np.random.default_rng(preset_seed)

        z_low, z_high = shift_range["z"]
        yx_low, yx_high = shift_range["yx"]
        z_options = [v for v in range(z_low, z_high + 1) if v != 0]
        z_shift = int(rng_shift.choice(z_options)) if z_options else int(rng_shift.integers(z_low, z_high + 1))
        y_shift = int(rng_shift.integers(yx_low, yx_high + 1))
        x_shift = int(rng_shift.integers(yx_low, yx_high + 1))
        shift = (z_shift, y_shift, x_shift)

        shifted_spots = _apply_shift_to_spots(spot_positions, shift, shape)
        mov_shift = render_spots(
            shape=shape,
            spots=_scene_table(shifted_spots),
            background=20,
            noise_std=5,
            seed=seed + 1,
            add_noise=add_noise,
        )
        rounds['shift'] = mov_shift
        truth_rows.extend(_truth_rows(spot_positions, namespace=preset, round_label='shift', shape=shape, shift=shift))
        effective_seeds['shift'] = preset_seed
        perturbations['shift'] = {'shift_zyx': shift, 'direction': 'reference_to_moving', 'units': 'voxel_index'}
        ground_truth["pairs"]["shift"] = {
            "type": "global_shift",
            "shift_zyx": list(shift),
        }

        # Generate deformed moving images
        for deform_name, deform_config in DEFORMATION_CONFIGS.items():
            scaled_config = _scale_deformation_config(deform_config, shape)

            deform_field = generate_displacement_field(
                shape=shape,
                deform_type=scaled_config["type"],
                max_displacement=scaled_config["max_displacement"],
                seed=seed + hash(deform_name) % 10000,
                **{k: v for k, v in scaled_config.items() if k not in ["type", "max_displacement"]},
            )

            deformed_spots = _apply_deformation_to_spots(spot_positions, deform_field, shape)
            mov_deform = render_spots(
                shape=shape,
                spots=_scene_table(deformed_spots),
                background=20,
                noise_std=5,
                seed=seed + 2,
                add_noise=add_noise,
            )

            rounds[deform_name] = mov_deform
            truth_rows.extend(_truth_rows(spot_positions, namespace=preset, round_label=deform_name, shape=shape, field=deform_field))
            effective_seeds[deform_name] = seed + hash(deform_name) % 10000
            perturbations[deform_name] = {'field': deform_field, 'direction': 'reference_to_moving',
                'representation': 'forward_displacement_at_integer_center', 'units': 'voxel_index'}
            ground_truth["pairs"][deform_name] = {
                "type": "local_deformation",
                "deformation_type": deform_name,
                "max_displacement": round(scaled_config["max_displacement"], 1),
                "field_file": f"field_{deform_name}.npy",
            }

        metadata = {label: ImageMetadata(f'{preset}/{label}') for label in rounds}
        results[preset] = SyntheticDataset({preset: rounds}, {preset: metadata}, ('ch00',),
            _truth_table(truth_rows), None,
            {'preset': preset, 'shape_zyx': shape, 'n_spots': n_spots, 'seed': seed, 'add_noise': add_noise,
             'shift_range': deepcopy(shift_range), 'deformations': deepcopy(DEFORMATION_CONFIGS)},
            {'seed': seed, 'effective_seeds': effective_seeds, 'limitations': LIMITATIONS},
            perturbations, ground_truth)
    return results
