"""Synthetic generation configuration."""
from dataclasses import dataclass
from typing import Literal

@dataclass
class SyntheticConfig:
    """Configuration for synthetic dataset generation.

    Parameters
    ----------
    height : int
        Y pixels. Default 256.
    width : int
        X pixels. Default 256.
    n_z : int
        Z slices. Default 10.
    n_fovs : int
        Number of FOVs. Default 2.
    n_rounds : int
        Sequencing round count. Default 4.
    n_channels : int
        Channel count, normally four for the color codebook. Default 4.
    n_spots_per_fov : int
        Number of sampled spots per FOV. Default 50.
    spot_sigma : float
        Gaussian spot width in voxel units. Default 1.5.
    spot_intensity : tuple[int, int]
        Inclusive minimum/maximum sampled peak intensities. Default (200, 255).
    background_mean : int
        Constant rendering background intensity. Default 20.
    background_std : int
        Compatibility field; currently unused by dataset generation. Default 5.
    noise_std : int
        Gaussian noise standard deviation in intensity units. Default 10.
    add_noise : bool
        Whether to add noise. Default True.
    dtype : Literal['uint8', 'uint16']
        Output uint8 or uint16. Default 'uint8'.
    max_shift_xy : int
        Maximum sampled X/Y translation magnitude in voxels. Default 5.
    max_shift_z : int
        Maximum sampled Z translation magnitude in voxels. Default 2.
    seed : int
        Random seed. Default 42.
    codebook : list[tuple[str, str]] | None
        List of (gene, nucleotide barcode) pairs; None uses _TEST_CODEBOOK. Default None.
    deformation : str | None
        DEFORMATION_CONFIGS name for non-reference rounds; None disables deformation. Default None.

    """

    # Image dimensions
    height: int = 256
    width: int = 256
    n_z: int = 10

    # Dataset structure
    n_fovs: int = 2
    n_rounds: int = 4
    n_channels: int = 4

    # Spot generation
    n_spots_per_fov: int = 50
    spot_sigma: float = 1.5
    spot_intensity: tuple[int, int] = (200, 255)

    # Noise and background
    background_mean: int = 20
    background_std: int = 5
    noise_std: int = 10
    add_noise: bool = True

    # Output dtype
    dtype: Literal["uint8", "uint16"] = "uint8"

    # Registration shifts (for testing registration)
    max_shift_xy: int = 5
    max_shift_z: int = 2

    # Random seed for reproducibility
    seed: int = 42

    # Custom codebook (None = use _TEST_CODEBOOK)
    codebook: list[tuple[str, str]] | None = None

    # Deformation (optional, for rounds 2+)
    # None = no deformation, or a key from DEFORMATION_CONFIGS
    deformation: str | None = None

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        """Volume shape in ZYX order."""
        return (self.n_z, self.height, self.width)
