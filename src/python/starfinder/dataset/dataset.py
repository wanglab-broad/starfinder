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
    channel_order labels the sequencing rounds and, by default, the other
    rounds; other_channel_order gives an other round its own labels (for
    example a morphology round's), in its C order.
    The repr summarizes IDs, round and channel labels, codebook size and roots.
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

    # Channel labels of other rounds that have their own channels
    other_channel_order: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self):
        self.rounds.validate()
        self.channel_order = tuple(self.channel_order)
        if len(set(self.channel_order)) != len(self.channel_order):
            raise ValueError("channel_order must be unique")
        self.other_channel_order = {name: tuple(labels) for name, labels in dict(self.other_channel_order).items()}
        for name, labels in self.other_channel_order.items():
            if name not in self.rounds.other_rounds:
                raise ValueError(f"other_channel_order round {name!r} is not an other round")
            if not labels or len(set(labels)) != len(labels) or any(not isinstance(x, str) or not x for x in labels):
                raise ValueError(f"channel labels of round {name!r} must be nonempty and unique")

    def channel_labels(self, round_name: str) -> tuple[str, ...]:
        """Channel labels of one round, in its C order.

        Parameters
        ----------
        round_name : str
            A configured round.

        Returns
        -------
        tuple[str, ...]
            other_channel_order[round_name] for an other round listed there,
            else channel_order.
        """
        if round_name not in self.rounds.all_rounds:
            raise ValueError(f"round {round_name!r} is not a configured round")
        return self.other_channel_order.get(round_name, self.channel_order)

    def __repr__(self):
        ref = self.rounds.reference_round

        def names(rounds):
            return ", ".join(r + "*" if r == ref else r for r in rounds) or "none"

        note = "   (* reference)" if ref is not None else ""
        codebook = (
            "not loaded" if self.codebook is None
            else f"{self.codebook.n_genes} genes × {len(self.codebook.round_labels)} rounds"
        )
        return "\n".join([
            f"Dataset {self.dataset_id!r} (sample {self.sample_id!r}, output {self.output_id!r})",
            f"    sequencing rounds: {names(self.rounds.sequencing_rounds)}"
            + (note if ref in self.rounds.sequencing_rounds else ""),
            f"    other rounds:      {names(self.rounds.other_rounds)}"
            + (note if ref in self.rounds.other_rounds else ""),
            f"    channels:          {', '.join(self.channel_order) or 'none'}",
            f"    codebook:          {codebook}",
            f"    input root:        {self.input_root}",
            f"    output root:       {self.output_root}",
        ])

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
