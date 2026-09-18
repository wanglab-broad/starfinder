"""Explicit historical measurement adapter retained for contract tests."""
import numpy as np
from starfinder.image import ImageMetadata
from starfinder.spot_finding import find_spots, PercentileCentroidConfig

def _evaluate_images(ref, mov_before, registered, use_mip=False):
    """Benchmark adapter: explicitly prepare legacy detections, then evaluate.

    Historical percentile choices remain here, outside pure evaluation. Volume
    NCC is retained for both policies. Constant reference range uses explicit 1.
    """
    from dataclasses import asdict
    from starfinder.evaluation.registration import evaluate_registration

    images = [ref, mov_before, registered]
    domain = [x.max(axis=0)[None, ...] for x in images] if use_mip else images
    percentile = 99.0 if use_mip else 99.5
    metadata = ImageMetadata("benchmark/reference-grid")
    spots = [find_spots(x, config=PercentileCentroidConfig(99.5),
                       metadata=metadata, spot_namespace=f"evaluation/{i}")
             .spots[["z", "y", "x"]].to_numpy() for i, x in enumerate(domain)]
    masks = [x > np.percentile(x, percentile) for x in domain]
    data_range = float(np.ptp(domain[0])) or 1.0
    report = evaluate_registration(
        *images, reference_spots=spots[0], before_spots=spots[1], after_spots=spots[2],
        reference_mask=masks[0], before_mask=masks[1], after_mask=masks[2],
        reference_metadata=metadata, before_metadata=metadata, after_metadata=metadata,
        data_range=data_range, ssim_policy="mip" if use_mip else "volume",
        matching_policy="greedy", match_threshold=2.0, units="voxel")
    return {**report.values,
            **{k: v for k, v in report.counts.items() if k.startswith("n_spots")},
            "ssim_method": "mip" if use_mip else "3d",
            "spot_method": "mip" if use_mip else "3d",
            "evaluation": asdict(report),
            "detection_config": {"percentile": 99.5, "mask_percentile": percentile, "method": "percentile_centroid"}}


