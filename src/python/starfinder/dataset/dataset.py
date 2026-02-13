"""STARMapDataset: sample-level configuration and FOV factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from starfinder.dataset.types import (
    ChannelOrder,
    Codebook,
    LayerState,
    SubtileConfig,
)

if TYPE_CHECKING:
    from starfinder.dataset.fov import FOV


@dataclass
class STARMapDataset:
    """Sample-level configuration and FOV factory.

    Non-frozen: allows lazy loading of codebook and subtile config.
    FOVs access dataset-level state via delegation properties.
    """

    # Paths
    input_root: Path  # {root_input_path}/{dataset_id}/{sample_id}
    output_root: Path  # {root_output_path}/{dataset_id}/{output_id}

    # Sample metadata
    dataset_id: str
    sample_id: str
    output_id: str

    # Dataset-level state (shared across FOVs)
    layers: LayerState = field(default_factory=LayerState)
    channel_order: ChannelOrder = field(default_factory=list)
    codebook: Codebook | None = None
    subtile: SubtileConfig | None = None

    # Processing parameters
    rotate_angle: float = 0.0
    maximum_projection: bool = False
    fov_pattern: str = "Position%03d"

    @classmethod
    def from_config(cls, config: dict) -> STARMapDataset:
        """Create dataset from validated Snakemake config dict."""
        layers = LayerState(
            seq=[f"round{i}" for i in range(1, config["n_rounds"] + 1)],
            ref=config["ref_round"],
        )
        return cls(
            input_root=Path(config["root_input_path"])
            / config["dataset_id"]
            / config["sample_id"],
            output_root=Path(config["root_output_path"])
            / config["dataset_id"]
            / config["output_id"],
            dataset_id=config["dataset_id"],
            sample_id=config["sample_id"],
            output_id=config["output_id"],
            layers=layers,
            channel_order=config.get("channel_order", []),
            rotate_angle=config.get("rotate_angle", 0.0),
            maximum_projection=config.get("maximum_projection", False),
            fov_pattern=config["fov_id_pattern"],
        )

    def fov(self, fov_id: str) -> FOV:
        """Create a new FOV instance for processing."""
        from starfinder.dataset.fov import FOV

        return FOV(dataset=self, fov_id=fov_id)

    def fov_ids(self, n_fovs: int, start: int = 0) -> list[str]:
        """Generate FOV ID list based on pattern."""
        return [self.fov_pattern % i for i in range(start, start + n_fovs)]

    def load_codebook(
        self,
        path: Path | str,
        split_index: int | None = None,
        do_reverse: bool = True,
    ) -> None:
        """Load codebook from CSV and store on self.codebook."""
        self.codebook = Codebook.from_csv(
            path, do_reverse=do_reverse, split_index=split_index
        )
