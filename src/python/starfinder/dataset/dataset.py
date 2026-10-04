"""Dataset: sample-level configuration and FOV factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from starfinder.barcode import (BarcodeLayout, Codebook, DirectPanel, EncodingConfig, OneBaseEncodingConfig,
    load_codebook, load_direct_panel)
from starfinder.barcode.codebook import _encoding_summary
from starfinder.barcode.decoding import READOUT_MODES
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
    readout_mode is how reads get their identity (docs/readout-contract.md,
    "Readout modes"): ``multiplexed`` (default) decodes color sequences with
    the codebook; ``direct`` assigns each candidate the direct_panel gene of its
    own round and channel.
    The repr summarizes IDs, round and channel labels, the reference (codebook
    with its encoding method and segment layout, or the panel in direct mode)
    and roots.
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

    # Readout mode and the direct-readout reference
    readout_mode: str = "multiplexed"
    direct_panel: DirectPanel | None = None

    def __post_init__(self):
        if self.readout_mode not in READOUT_MODES:
            raise ValueError(f"readout_mode must be one of {READOUT_MODES}; got {self.readout_mode!r}")
        if self.direct_panel is not None and not isinstance(self.direct_panel, DirectPanel):
            raise TypeError("direct_panel must be a DirectPanel or None")
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
        reference = [f"    codebook:          {codebook}"]
        if self.codebook is not None:
            reference.append(f"    encoding:          {_encoding_summary(self.codebook)}")
        if self.readout_mode == "direct":
            panel = ("not loaded" if self.direct_panel is None else
                     f"{self.direct_panel.n_genes} genes over {len(self.direct_panel.round_labels)} rounds")
            reference = ["    readout mode:      direct", f"    direct panel:      {panel}"]
        return "\n".join([
            f"Dataset {self.dataset_id!r} (sample {self.sample_id!r}, output {self.output_id!r})",
            f"    sequencing rounds: {names(self.rounds.sequencing_rounds)}"
            + (note if ref in self.rounds.sequencing_rounds else ""),
            f"    other rounds:      {names(self.rounds.other_rounds)}"
            + (note if ref in self.rounds.other_rounds else ""),
            f"    channels:          {', '.join(self.channel_order) or 'none'}",
            *reference,
            f"    input root:        {self.input_root}",
            f"    output root:       {self.output_root}",
        ])

    def encoding_table(self) -> pd.DataFrame:
        """The barcode encoding table of the loaded codebook.

        Returns
        -------
        pandas.DataFrame
            :meth:`starfinder.barcode.Codebook.encoding_table`: columns bases,
            color and channel (the channel label of the color).

        Raises
        ------
        ValueError
            readout_mode is ``direct`` (no barcode encoding) or no codebook is
            loaded.
        """
        if self.readout_mode == "direct":
            raise ValueError("readout_mode='direct' has no barcode encoding: reads are assigned from the "
                             "direct panel, so there is no encoding table")
        if self.codebook is None:
            raise ValueError("no codebook is loaded, so there is no encoding table; call "
                             "dataset.load_codebook() first")
        return self.codebook.encoding_table()

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
        *,
        encoding: EncodingConfig | OneBaseEncodingConfig | None = None,
        layout: BarcodeLayout | None = None,
    ) -> None:
        """Load codebook from CSV and store on self.codebook.

        Parameters
        ----------
        path : pathlib.Path or str
            Two-column gene,barcode CSV, with or without a header, or a
            canonical entry_id,gene_id,color_sequence[,base_sequence] CSV.
        split_index : int or None
            Optional zero-based two-segment split of the two_base encoding,
            default None. MATLAB's one-based split_index s is s - 1 here.
        reverse_bases : bool
            Reverse bases before encoding, default True.
        encoding : EncodingConfig, OneBaseEncodingConfig or None
            The encoding config; replaces split_index and reverse_bases, which
            must then keep their defaults.
        layout : BarcodeLayout or None
            Segment layout; None is one segment (or the legacy split).

        Returns
        -------
        None
            Stores the canonical barcode Codebook. Sequencing round labels and
            channel_order must be configured explicitly. Errors propagate from
            :func:`starfinder.barcode.load_codebook`.
        """
        if encoding is None:
            encoding = EncodingConfig(reverse_bases=reverse_bases, split_index=split_index)
        elif split_index is not None or reverse_bases is not True:
            raise ValueError("give encoding, or split_index and reverse_bases, not both")
        self.codebook = load_codebook(path, round_labels=tuple(self.rounds.sequencing_rounds),
            channel_labels=tuple(self.channel_order), encoding=encoding, layout=layout)

    def load_direct_panel(self, path: Path | str) -> None:
        """Load the direct-readout panel from CSV and store it on self.direct_panel.

        Parameters
        ----------
        path : pathlib.Path or str
            CSV with the header round,channel,gene_id: the gene of each
            (round, channel). Rounds must be sequencing rounds and channels
            labels of channel_order.

        Returns
        -------
        None
            Stores the DirectPanel. Errors propagate from
            :func:`starfinder.barcode.load_direct_panel`.
        """
        self.direct_panel = load_direct_panel(path, round_labels=tuple(self.rounds.sequencing_rounds),
                                              channel_labels=tuple(self.channel_order))
