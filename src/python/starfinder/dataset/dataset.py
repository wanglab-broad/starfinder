"""Dataset: sample-level configuration and FOV factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from starfinder.barcode import Codebook, EncodingConfig, load_codebook
from starfinder.dataset.types import (
    RoundState,
    SubtileConfig,
)

if TYPE_CHECKING:
    from starfinder.dataset.fov import FOV


@dataclass
class Dataset:
    """Sample paths, ordered rounds/channel labels, codebook and FOV factory.

    Processing options belong to PipelineConfig, residency to ExecutionConfig.
    Shared workflow YAML is translated by from_workflow_config.
    """

    # Paths
    input_root: Path  # {root_input_path}/{dataset_id}/{sample_id}
    output_root: Path  # {root_output_path}/{dataset_id}/{output_id}

    # Sample metadata
    dataset_id: str
    sample_id: str
    output_id: str

    # Dataset-level state (shared across FOVs)
    rounds: RoundState = field(default_factory=RoundState)
    channel_order: tuple[str, ...] = ()
    codebook: Codebook | None = None
    subtile: SubtileConfig | None = None

    # Processing parameters
    fov_pattern: str = "Position%03d"

    def __post_init__(self):
        self.rounds.validate()
        self.channel_order = tuple(self.channel_order)
        if len(set(self.channel_order)) != len(self.channel_order):
            raise ValueError("channel_order must be unique")

    def fov(self, fov_id: str) -> FOV:
        """Create a new FOV instance for processing.

        Parameters
        ----------
        fov_id : str
            FOV identifier, used in paths.

        Returns
        -------
        FOV
            New empty processor sharing this dataset, without loading images.
        """
        from starfinder.dataset.fov import FOV

        return FOV(dataset=self, fov_id=fov_id)

    def fov_ids(self, n_fovs: int, start: int = 0) -> list[str]:
        """Generate FOV ID list based on pattern.

        Parameters
        ----------
        n_fovs : int
            Number of names to generate.
        start : int
            First numeric FOV index, default 0.

        Returns
        -------
        list[str]
            fov_pattern percent-formatted with start through start+n_fovs-1.
        """
        return [self.fov_pattern % i for i in range(start, start + n_fovs)]

    def load_codebook(
        self,
        path: Path | str,
        split_index: int | None = None,
        reverse_bases: bool = True,
    ) -> None:
        """Load codebook from CSV and store on self.codebook.

        Parameters
        ----------
        path : pathlib.Path or str
            Two-column gene,barcode CSV, with or without a header.
        reverse_bases : bool
            Reverse bases before encoding, default True.
        split_index : int or None
            Optional two-segment split, default None.

        Returns
        -------
        None
            Stores the canonical barcode Codebook. Sequencing round labels and
            channel_order must be configured explicitly. Errors propagate from
            :func:`starfinder.barcode.load_codebook`.
        """
        self.codebook = load_codebook(path, round_labels=tuple(self.rounds.sequencing_rounds),
            channel_labels=tuple(self.channel_order),
            encoding=EncodingConfig(reverse_bases=reverse_bases, split_index=split_index))
