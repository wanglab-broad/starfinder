"""Private historical preset values; dataset and registration shifts differ."""
from itertools import product
from typing import Literal
from starfinder.barcode import EncodingConfig
from ._config import SyntheticConfig

_TEST_CODEBOOK = [
    ("GeneA", "CACGC"),
    ("GeneB", "CATGC"),
    ("GeneC", "CGAAC"),
    ("GeneD", "CGTAC"),
    ("GeneE", "CTGAC"),
    ("GeneF", "CTAGC"),
    ("GeneG", "CCATC"),
    ("GeneH", "CGCTC"),
]


# Standard volume size presets (Z, Y, X)
#: Registration preset names to volume shapes (Z, Y, X) in voxels.
SIZE_PRESETS: dict[str, tuple[int, int, int]] = {
    "tiny": (8, 128, 128),
    "small": (16, 256, 256),
    "medium": (32, 512, 512),
    "large": (30, 1024, 1024),
    "tissue": (30, 3072, 3072),        # tissue-2D size
    "thick_medium": (100, 1024, 1024),  # thick tissue, medium XY
}

# Spot density: approximately 50 spots per 10^6 voxels
#: Registration preset names to synthetic spot counts.
SPOT_COUNTS: dict[str, int] = {
    "tiny": 10,
    "small": 50,
    "medium": 400,
    "large": 1500,
    "tissue": 14000,
    "thick_medium": 5200,
}

# Shift ranges for global registration testing (≤25% of each dimension)
#: Registration presets to inclusive z and shared yx shift ranges (low, high), in voxels.
SHIFT_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "tiny": {"z": (-2, 2), "yx": (-10, 10)},
    "small": {"z": (-4, 4), "yx": (-25, 25)},
    "medium": {"z": (-8, 8), "yx": (-50, 50)},
    "large": {"z": (-7, 7), "yx": (-100, 100)},
    "tissue": {"z": (-7, 7), "yx": (-300, 300)},
    "thick_medium": {"z": (-25, 25), "yx": (-100, 100)},
}


DEFORMATION_CONFIGS = {
    "polynomial_small": {"type": "polynomial", "max_displacement_pct": 3.0, "cap_px": 15.0},
    "polynomial_large": {"type": "polynomial", "max_displacement_pct": 6.0, "cap_px": 30.0},
    "gaussian_small": {"type": "gaussian", "max_displacement_pct": 3.0, "radius_pct": 6.0, "cap_px": 15.0},
    "gaussian_large": {"type": "gaussian", "max_displacement_pct": 6.0, "radius_pct": 10.0, "cap_px": 30.0},
    "multi_point": {"type": "multi_point", "max_displacement_pct": 4.0, "n_points": 4, "radius_pct": 5.0, "cap_px": 20.0},
    "linear_small": {"type": "linear", "max_displacement_pct": 1.0, "cap_px": 10.0},
}


def generate_codebook(n_genes: int) -> list[tuple[str, str]]:
    """Generate a codebook with n_genes CNNNNC barcodes and unique color sequences.

    Enumerates all 5-base barcodes of the form C-{A,C,G,T}^3-C,
    filters to those with unique color sequences, and returns the
    first n_genes entries.

    Parameters
    ----------
    n_genes : int
        Number of genes (max 64 for 4-round, 4-channel).

    Returns
    -------
    list[tuple[str, str]]
        List of (gene_name, barcode) tuples.

    Raises
    ------
    ValueError
        If n_genes exceeds the number of unique color sequences.
    """
    bases = "ACGT"
    all_barcodes = [f"C{''.join(mid)}C" for mid in product(bases, repeat=3)]

    # Filter to unique color sequences
    seen_colors: dict[str, str] = {}
    unique_entries: list[tuple[str, str]] = []
    for barcode in all_barcodes:
        color_seq = EncodingConfig(reverse_bases=True).encode(barcode)
        if color_seq not in seen_colors:
            seen_colors[color_seq] = barcode
            unique_entries.append((barcode, color_seq))

    if n_genes > len(unique_entries):
        raise ValueError(
            f"Requested {n_genes} genes but only {len(unique_entries)} "
            f"unique color sequences available with CNNNNC barcodes."
        )

    codebook = []
    for i, (barcode, _) in enumerate(unique_entries[:n_genes]):
        gene_name = f"Gene{i + 1:03d}"
        codebook.append((gene_name, barcode))

    return codebook
def _scale_deformation_config(config: dict, shape: tuple[int, int, int]) -> dict:
    """Scale deformation config from percentages to absolute pixels.

    Parameters
    ----------
    config : dict
        Deformation config with _pct suffix fields and optional cap_px.
    shape : tuple[int, int, int]
        Volume shape (Z, Y, X).

    Returns
    -------
    dict
        Config with absolute pixel values.
    """
    min_xy = min(shape[1], shape[2])
    scaled = {"type": config["type"]}

    if "max_displacement_pct" in config:
        displacement = config["max_displacement_pct"] * min_xy / 100.0
        # Apply cap if specified
        if "cap_px" in config:
            displacement = min(displacement, config["cap_px"])
        scaled["max_displacement"] = displacement

    if "radius_pct" in config:
        scaled["radius"] = config["radius_pct"] * min_xy / 100.0

    if "n_points" in config:
        scaled["n_points"] = config["n_points"]

    return scaled
def get_preset_config(
    preset: Literal["tiny", "small", "medium", "large", "tissue", "thick_medium"],
) -> SyntheticConfig:
    """Get predefined configuration for a preset.

    Parameters
    ----------
    preset : {"tiny", "small", "medium", "large", "tissue", "thick_medium"}

        - "tiny": 2 FOVs, 128x128x8, 10 spots (quick tests)

        - "small": 2 FOVs, 256x256x16, 50 spots (unit tests)

        - "medium": 2 FOVs, 512x512x32, 400 spots (integration tests)

        - "large": 2 FOVs, 1024x1024x30, 1500 spots (e2e benchmarking)

        - "tissue": 2 FOVs, 3072x3072x30, 14000 spots (tissue-2D scale)

        - "thick_medium": 2 FOVs, 1024x1024x100, 5200 spots (thick tissue)

    Returns
    -------
    SyntheticConfig
        Configuration for the specified preset.
    """
    presets = {
        "tiny": SyntheticConfig(
            height=128,
            width=128,
            n_z=8,
            n_fovs=2,
            n_spots_per_fov=10,
            max_shift_xy=5,
            max_shift_z=2,
            seed=42,
        ),
        "small": SyntheticConfig(
            height=256,
            width=256,
            n_z=16,
            n_fovs=2,
            n_spots_per_fov=50,
            seed=42,
        ),
        "medium": SyntheticConfig(
            height=512,
            width=512,
            n_z=32,
            n_fovs=2,
            n_spots_per_fov=400,
            max_shift_xy=50,
            max_shift_z=8,
            seed=42,
        ),
        "large": SyntheticConfig(
            height=1024,
            width=1024,
            n_z=30,
            n_fovs=2,
            n_spots_per_fov=1500,
            max_shift_xy=100,
            max_shift_z=7,
            seed=123,
            codebook=generate_codebook(64),
        ),
        "tissue": SyntheticConfig(
            height=3072,
            width=3072,
            n_z=30,
            n_fovs=2,
            n_spots_per_fov=14000,
            max_shift_xy=300,
            max_shift_z=7,
            seed=456,
            codebook=generate_codebook(64),
        ),
        "thick_medium": SyntheticConfig(
            height=1024,
            width=1024,
            n_z=100,
            n_fovs=2,
            n_spots_per_fov=5200,
            max_shift_xy=100,
            max_shift_z=25,
            seed=789,
            codebook=generate_codebook(64),
        ),
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from: {list(presets.keys())}")
    return presets[preset]
