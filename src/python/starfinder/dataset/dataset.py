"""Dataset: sample-level configuration and FOV factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from starfinder.barcode import (BarcodeLayout, Codebook, DirectPanel, EncodingConfig, OneBaseEncodingConfig,
    load_codebook, load_direct_panel)
from starfinder.barcode.codebook import _encoding_summary
from starfinder.barcode.decoding import READOUT_MODES
from starfinder.dataset.types import (
    REFERENCE_STAIN,
    ChannelInfo,
    RoundState,
    SubtileConfig,
    _channel_infos,
)

if TYPE_CHECKING:
    from starfinder.dataset.fov import FOV


@dataclass
class Dataset:
    """Sample paths, ordered rounds/channels, codebook and FOV factory.

    Processing options belong to PipelineConfig, residency to ExecutionConfig.
    Shared workflow YAML is translated by from_workflow_config.
    The channels come in three groups (docs/coordination.md, "Channels"):
    channel_order, the sequencing colours, which also label the other rounds
    by default; other_channel_order, an other round's own channels (for
    example a morphology round's), in its C order; and reference_stains, the
    stain files of the reference round's folder that are not sequencing
    colours (image name ``reference_stain``). Each channel is given as its
    file pattern (a string), a :class:`ChannelInfo` or a mapping with the keys
    channel, name and wavelength; after construction channel_order and
    other_channel_order hold the patterns (the channel labels) and
    :meth:`channel_info` returns the full ChannelInfo of a round.
    With ``dataclasses.replace``, channels the call passes are read as
    given, except that the dataset's own tuple of a round (channel_order, or
    other_channel_order[round] inside any mapping), which replace passes
    for an unchanged field, keeps that round's ChannelInfo.
    :meth:`channel_index` is the one rule that finds a channel by index,
    pattern or name.
    readout_mode is how reads get their identity (docs/readout-contract.md,
    "Readout modes"): ``multiplexed`` (default) decodes color sequences with
    the codebook; ``direct`` assigns each candidate the direct_panel gene of its
    own round and channel.
    The repr summarizes IDs, round and channel labels, the reference (codebook
    with its encoding method and segment layout, or the panel in direct mode)
    and roots.

    Raises
    ------
    ValueError
        Invalid rounds, a round named ``reference_stain``, repeated patterns
        within a round, other_channel_order for a round that is not an other
        round or with no channel, reference stains without a reference round or
        sharing a pattern with channel_order, or the field errors of ChannelInfo.
    TypeError
        A channel that is not a pattern string, ChannelInfo or mapping.
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

    # Channels of other rounds that have their own channels
    other_channel_order: dict[str, tuple[str, ...]] = field(default_factory=dict)

    # Readout mode and the direct-readout reference
    readout_mode: str = "multiplexed"
    direct_panel: DirectPanel | None = None

    # Stain files of the reference round's folder that are not sequencing colours
    reference_stains: tuple[ChannelInfo, ...] = ()

    # The patterns and ChannelInfo of the sequencing channels (key None) and of each other round with its own
    # channels. An init field so that dataclasses.replace hands it to the copy, which keeps the ChannelInfo of
    # every round whose patterns the call leaves as they are (__post_init__).
    _channels: dict = field(default_factory=dict, repr=False, kw_only=True)

    def __post_init__(self):
        if self.readout_mode not in READOUT_MODES:
            raise ValueError(f"readout_mode must be one of {READOUT_MODES}; got {self.readout_mode!r}")
        if self.direct_panel is not None and not isinstance(self.direct_panel, DirectPanel):
            raise TypeError("direct_panel must be a DirectPanel or None")
        self.rounds.validate()
        if REFERENCE_STAIN in self.rounds.all_rounds:
            raise ValueError(f"{REFERENCE_STAIN!r} is the reserved image name of the reference stains; "
                             "no configured round may take it")
        kept = dict(self._channels)

        def infos(key, channels, what):
            # dataclasses.replace passes the source's own pattern tuple for a round the call does not change; that
            # very tuple keeps the source's ChannelInfo, while any other value is read as given.
            if key in kept and channels is kept[key][0]:
                return kept[key][1]
            return _channel_infos(channels, what)

        sequencing = infos(None, self.channel_order, "channel_order")
        others = {}
        for name, channels in dict(self.other_channel_order).items():
            if name not in self.rounds.other_rounds:
                raise ValueError(f"other_channel_order round {name!r} is not an other round")
            others[name] = infos(name, channels, f"the channels of round {name!r}")
            if not others[name]:
                raise ValueError(f"channel labels of round {name!r} must be nonempty and unique")
        self.reference_stains = _channel_infos(self.reference_stains, "reference_stains")
        if self.reference_stains:
            if self.rounds.reference_round is None:
                raise ValueError("reference_stains are files of the reference round, but no reference round is set")
            shared = sorted({c.channel for c in self.reference_stains} & {c.channel for c in sequencing})
            if shared:
                raise ValueError(f"reference_stains share the patterns {shared} with channel_order; a reference "
                                 "stain is not a sequencing colour")
        self.channel_order = tuple(c.channel for c in sequencing)
        self.other_channel_order = {name: tuple(c.channel for c in channels) for name, channels in others.items()}
        self._channels = {None: (self.channel_order, sequencing),
                          **{name: (self.other_channel_order[name], channels) for name, channels in others.items()}}

    def channel_info(self, round_name: str) -> tuple[ChannelInfo, ...]:
        """The channels of one round, in its C order, with pattern, name and wavelength.

        Parameters
        ----------
        round_name : str
            A configured round, or ``reference_stain`` for the reference stains.

        Returns
        -------
        tuple[ChannelInfo, ...]
            The other round's own channels when other_channel_order lists it,
            reference_stains for ``reference_stain``, else the sequencing
            channels (channel_order).

        Raises
        ------
        ValueError
            round_name is not a configured round or ``reference_stain``.
        """
        if round_name == REFERENCE_STAIN:
            return self.reference_stains
        if round_name not in self.rounds.all_rounds:
            raise ValueError(f"round {round_name!r} is not a configured round")
        if round_name in self.other_channel_order:
            return self._channels_of(round_name, self.other_channel_order[round_name])
        return self._channels_of(None, self.channel_order)

    def channel_labels(self, round_name: str) -> tuple[str, ...]:
        """Channel labels (file patterns) of one round, in its C order.

        Parameters
        ----------
        round_name : str
            A configured round, or ``reference_stain``.

        Returns
        -------
        tuple[str, ...]
            other_channel_order[round_name] for an other round listed there,
            the reference stains' patterns for ``reference_stain``, else
            channel_order.
        """
        return tuple(c.channel for c in self.channel_info(round_name))

    def channel_index(self, round_name: str, key: str | int) -> int:
        """The C index of one channel of a round: the one channel lookup rule.

        An integer is an index. A string is first an exact channel pattern;
        otherwise an exact name that occurs once in the round. So a string that
        is the pattern of one channel and the name of another resolves as the
        pattern.

        Parameters
        ----------
        round_name : str
            A configured round, or ``reference_stain``.
        key : str or int
            A channel pattern, a channel name or a nonnegative index.

        Returns
        -------
        int
            The channel's position in channel_info(round_name).

        Raises
        ------
        ValueError
            An unknown round, a name that occurs more than once, a key that is
            neither a pattern nor a name of the round, or an index outside the
            round; the message names the round and its channels.
        TypeError
            key is not a string or an integer.
        """
        return self._index(round_name, self.channel_info(round_name), key)

    def _resident_channels(self, round_name):
        """channel_info of a resident image; an image under a name the dataset does not configure takes the
        sequencing channels, as FOV.register_rounds and FOV.segment did before channel_index."""
        configured = round_name == REFERENCE_STAIN or round_name in self.rounds.all_rounds
        return self.channel_info(round_name) if configured else self._channels_of(None, self.channel_order)

    def _resident_channel_index(self, round_name, key):
        """channel_index on the channels of a resident image (_resident_channels)."""
        return self._index(round_name, self._resident_channels(round_name), key)

    def _channels_of(self, key, labels):
        """The stored ChannelInfo of key (None: the sequencing channels); patterns assigned after construction carry
        no name or wavelength."""
        channels = self._channels.get(key, ((), ()))[1]
        if tuple(c.channel for c in channels) != tuple(labels):
            channels = tuple(ChannelInfo(label) for label in labels)
        return channels

    @staticmethod
    def _index(round_name, channels, key):
        """The lookup rule of channel_index on the given channels of round_name."""
        listed = ", ".join(c.channel if c.name is None else f"{c.channel} ({c.name})" for c in channels) or "none"
        if isinstance(key, bool) or not isinstance(key, (str, Integral)):
            raise TypeError(f"a channel key is a pattern, a name or an index; got {key!r}")
        if isinstance(key, Integral):
            if not 0 <= key < len(channels):
                raise ValueError(f"channel index {key} is outside round {round_name!r}; its channels are {listed}")
            return int(key)
        patterns = [c.channel for c in channels]
        if key in patterns:
            return patterns.index(key)
        named = [i for i, c in enumerate(channels) if c.name == key]
        if len(named) > 1:
            raise ValueError(f"round {round_name!r} has {len(named)} channels named {key!r}; name one by its "
                             f"pattern or index; its channels are {listed}")
        if not named:
            raise ValueError(f"round {round_name!r} has no channel {key!r}: {key!r} is not a channel label or a "
                             f"channel name of the round; its channels are {listed}")
        return named[0]

    def channel_record(self) -> dict[str, list[dict]]:
        """The written channel information: per configured round (and ``reference_stain`` when set) its channels.

        Returns
        -------
        dict[str, list[dict]]
            Round name to :meth:`ChannelInfo.record` entries (channel, name,
            wavelength; a missing wavelength is ``"unavailable"``), in C order.
        """
        names = [*self.rounds.all_rounds, *([REFERENCE_STAIN] if self.reference_stains else [])]
        return {name: [c.record() for c in self.channel_info(name)] for name in names}

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
