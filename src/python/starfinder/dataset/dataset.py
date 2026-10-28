"""Dataset: sample-level configuration and FOV factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from starfinder._layout import IMAGE_DIR_TEMPLATE, REFERENCE_IMAGE
from starfinder._records import file_sha256
from starfinder.readout import (BarcodeLayout, Codebook, DirectPanel, EncodingConfig, OneBaseEncodingConfig,
    load_codebook, load_direct_panel)
from starfinder.readout.codebook import _encoding_summary
from starfinder.readout.decoding import READOUT_MODES
from starfinder.dataset.types import (
    REFERENCE_STAIN,
    ChannelInfo,
    RoundState,
    SubtileConfig,
    _channel_infos,
)

if TYPE_CHECKING:
    from starfinder.dataset.fov import FOV


def _check_fov_pattern(pattern):
    """The ``{i}`` pattern of FOV names: ``str.format`` with the one field ``i``, such as ``Position{i:03d}``."""
    try:
        names = {pattern.format(i=0), pattern.format(i=1)} if isinstance(pattern, str) else set()
    except (KeyError, IndexError, ValueError, TypeError, AttributeError):
        names = set()
    if len(names) != 2:
        raise ValueError(f"fov_pattern must be a {{i}} pattern such as 'Position{{i:03d}}' (str.format with the field "
                         f"i; percent patterns are not read); got {pattern!r}")
    return pattern


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
    readout_mode is how reads get their identity (design/specs/readout.md,
    "Readout modes"): ``multiplexed`` (default) decodes color sequences with
    the codebook; ``direct`` assigns each candidate the direct_panel gene of its
    own round and channel.
    voxel_size_xy and voxel_size_z (µm, the workflow keys of the same names)
    declare the calibration of saved files that store none: ``stitch`` and
    ``export`` take a FOV's grid from the stored metadata of its reference
    image, or else from these sizes. reference_image (the workflow key of the
    same name) is the file pattern of that image, with ``{fov_id}``, absolute
    or relative to the output root; the default is
    ``fovs/{fov_id}/reference.tif``, the file ``FOV.save_reference_image``
    writes. reference_image_import (the workflow key of the same name; None by
    default) declares instead a reference image that another tool or rule
    produced, a pattern of the same form: every reader reads it, and
    ``FOV.save_reference_image`` refuses to write it. It excludes a
    reference_image other than the default.
    fov_pattern is the ``{i}`` pattern (``str.format`` with the field ``i``,
    default ``Position{i:03d}``) of :meth:`fov_ids` and of the FOV a stage
    file's ``PositionID`` names in :meth:`stitch`.
    image_dir_template (the workflow key of the same name) lays out the
    acquisition images under input_root: ``"{round}/{fov}"`` (default), one
    folder per round holding a folder per FOV, or ``"{fov}/{round}"``, one
    folder per FOV holding a folder per round. ``FOV.input_dir`` builds
    the folder a round's channel files are read from with it.
    The repr summarizes IDs, round and channel labels, the reference (codebook
    with its encoding method and segment layout, or the panel in direct mode)
    and roots.

    Raises
    ------
    ValueError
        Invalid rounds, a round named ``reference_stain``, repeated patterns
        within a round, other_channel_order for a round that is not an other
        round or with no channel, reference stains without a reference round or
        sharing a pattern with channel_order, a voxel size that is not a
        finite positive number or None, a reference_image or
        reference_image_import that is not a pattern with ``{fov_id}``, a
        fov_pattern that is not a ``{i}`` pattern, an image_dir_template other
        than the two layouts, both
        reference settings given, a channel pattern that is not a safe
        single path component (``[A-Za-z0-9][A-Za-z0-9_.-]*``; the Python
        workflow names channel folders by it), or the field errors of
        ChannelInfo.
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

    # The {i} pattern of fov_ids (str.format with the field i)
    fov_pattern: str = "Position{i:03d}"

    # Channels of other rounds that have their own channels
    other_channel_order: dict[str, tuple[str, ...]] = field(default_factory=dict)

    # Readout mode and the direct-readout reference
    readout_mode: str = "multiplexed"
    direct_panel: DirectPanel | None = None

    # Stain files of the reference round's folder that are not sequencing colours
    reference_stains: tuple[ChannelInfo, ...] = ()

    # The declared calibration (µm) of saved files that store none, as the workflow's voxel_size_xy and voxel_size_z
    voxel_size_xy: float | None = None
    voxel_size_z: float | None = None
    # The file pattern of a FOV's reference image, as the workflow's reference_image, or of an imported one, as the
    # workflow's reference_image_import, which every reader reads and nothing writes
    reference_image: str = REFERENCE_IMAGE
    reference_image_import: str | None = None
    # The layout of the acquisition images under input_root, as the workflow's image_dir_template: round-first
    # (default) or FOV-first
    image_dir_template: str = IMAGE_DIR_TEMPLATE

    # The patterns and ChannelInfo of the sequencing channels (key None) and of each other round with its own
    # channels. An init field so that dataclasses.replace hands it to the copy, which keeps the ChannelInfo of
    # every round whose patterns the call leaves as they are (__post_init__).
    _channels: dict = field(default_factory=dict, repr=False, kw_only=True)
    # The files load_codebook and load_direct_panel read, by kind (codebook, direct_panel): the loaded object,
    # the absolute path and the SHA-256 of its bytes; and the workflow configuration file from_workflow_config
    # translated. run.json records them (docs/checkpoints.md, "run.json").
    _reference_files: dict = field(default_factory=dict, repr=False, compare=False, kw_only=True)
    _workflow_config: Path | None = field(default=None, repr=False, compare=False, kw_only=True)
    # True on the workflow's legacy route (MATLAB's rsf_single_fov, or stitch_subtile in the subtile and deep
    # modes): export with assignment=None then reads the one-based signal/{fov_id}_goodSpots.csv.
    _legacy_molecules: bool = field(default=False, repr=False, compare=False, kw_only=True)

    def __post_init__(self):
        if self.readout_mode not in READOUT_MODES:
            raise ValueError(f"readout_mode must be one of {READOUT_MODES}; got {self.readout_mode!r}")
        if self.direct_panel is not None and not isinstance(self.direct_panel, DirectPanel):
            raise TypeError("direct_panel must be a DirectPanel or None")
        from starfinder._layout import check_image_dir_template, check_reference_image, check_reference_import
        from starfinder.dataset._declared import check_voxel_size
        self.voxel_size_xy = check_voxel_size("voxel_size_xy", self.voxel_size_xy)
        self.voxel_size_z = check_voxel_size("voxel_size_z", self.voxel_size_z)
        self.reference_image = check_reference_image(self.reference_image)
        self.reference_image_import = check_reference_import(self.reference_image, self.reference_image_import)
        self.fov_pattern = _check_fov_pattern(self.fov_pattern)
        self.image_dir_template = check_image_dir_template(self.image_dir_template)
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
        sequencing channels, as FOV.register and FOV.segment did before channel_index."""
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

    def _channel_record(self) -> dict[str, list[dict]]:
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
            :meth:`starfinder.readout.Codebook.encoding_table`: columns bases,
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
            ``fov_pattern.format(i=i)`` for i from start through
            start+n_fovs-1 (the workflow numbers FOVs from 1, so its callers
            pass ``start=1``).
        """
        return [self.fov_pattern.format(i=i) for i in range(start, start + n_fovs)]

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
            Stores the canonical barcode Codebook, and keeps the file's
            absolute path and SHA-256 for ``run.json`` ``codebook``. Sequencing
            round labels and channel_order must be configured explicitly.
            Errors propagate from :func:`starfinder.readout.load_codebook`.
        """
        if encoding is None:
            encoding = EncodingConfig(reverse_bases=reverse_bases, split_index=split_index)
        elif split_index is not None or reverse_bases is not True:
            raise ValueError("give encoding, or split_index and reverse_bases, not both")
        self.codebook = load_codebook(path, round_labels=tuple(self.rounds.sequencing_rounds),
            channel_labels=tuple(self.channel_order), encoding=encoding, layout=layout)
        self._keep_reference_file("codebook", self.codebook, path)

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
            Stores the DirectPanel, and keeps the file's absolute path and
            SHA-256 for ``run.json`` ``codebook``. Errors propagate from
            :func:`starfinder.readout.load_direct_panel`.
        """
        self.direct_panel = load_direct_panel(path, round_labels=tuple(self.rounds.sequencing_rounds),
                                              channel_labels=tuple(self.channel_order))
        self._keep_reference_file("direct_panel", self.direct_panel, path)

    def _keep_reference_file(self, kind, loaded, path):
        """Keep the path and SHA-256 of the file a codebook or panel was loaded from (a new dict: replace shares it)."""
        path = Path(path).absolute()
        sha256 = file_sha256(path)
        self._reference_files = {**self._reference_files, kind: (loaded, str(path), sha256)}

    def stitch(self, sample: str, fov_ids, *, table, config, input: str = "reference_merged", review=None,
               fallback: str | None = None, output_dir: Path | str | None = None, position_ids=None):
        """Place one sample's FOVs from their saved stitching images (design/specs/stitching.md).

        A sample-level computation, not a multi-FOV runner: it calls no per-FOV
        stage and reads only saved per-FOV files. For each FOV it reads the
        reference grid of its reference image, the file ``reference_image``
        names (default ``fovs/{fovID}/reference.tif``, the file
        ``FOV.save_reference_image`` writes) or ``reference_image_import``: its
        stored metadata, or for a
        file that stores none the grid ``voxel_size_xy`` and ``voxel_size_z``
        declare. It reads that grid and the stitching image named by
        ``input``, one FOV at a time, builds the
        :class:`~starfinder.stitching.StitchingInput` list in table order, calls
        :func:`~starfinder.stitching.stitch` and, with ``output_dir``, saves the
        result there with :func:`~starfinder.stitching.save_placement`.

        Parameters
        ----------
        sample : str
            The sample; the placement frame is ``sample:<sample>``.
        fov_ids : Sequence[str]
            The sample's FOVs; the table must hold exactly these.
        table : TilePositionTable or Mapping
            A built table, or the workflow's ``positions`` mapping: ``source``
            (``stage_file``, ``regular_grid`` or ``table``), ``stage_axes``
            (``{x, y}``, required with ``stage_file``), ``snap_to_grid``
            (``{overlap}`` or null), ``regular_grid`` (``{columns, rows, order,
            overlap}``), ``tile_shape_yx`` and ``spacing_yx`` (null: from the
            reference grids, which must then share them), ``table`` (a CSV path),
            ``unit`` (of the table, default ``µm``) and ``stage_file`` (the
            stage file's path, required with ``stage_file``; no default). The
            stage file may hold other samples' positions: the FOVs' positions
            are selected before the converter maps, shifts or snaps them
            (:func:`~starfinder.stitching.positions_from_leica`). A supplied
            table may hold other FOVs too: its rows of fov_ids are kept.
        config
            A config registered in ``STITCHING_METHODS``.
        input : str
            ``reference_merged`` (the reference image, default),
            ``reference_stain`` (the one-channel saved ``reference_stain`` of
            ``FOV.prepare_morphology`` under ``fovs/{fovID}/other_rounds/``)
            or ``file:<pattern>`` (``{fovID}`` replaced; relative to the output
            root), a file of the reference grid's shape. A missing input file
            gives the FOV an absent image (status ``missing``).
        review : ReviewPolicy or None
            The review criterion; None is ``ReviewPolicy()``.
        fallback : str or None
            None (default) or ``"nominal"``.
        output_dir : str, Path or None
            The sample's placement folder (the workflow passes
            ``<OUTPUT>/stitching/<sample>/``); None saves nothing.
        position_ids : Mapping[str, int] or None
            The stage file's ``PositionID`` of each FOV of fov_ids (the
            workflow passes the ``position_id`` column of ``fovs.csv``); None
            (default) names ``PositionID`` ``p`` the FOV
            ``fov_pattern.format(i=p)``, for a file that numbers its positions
            by FOV.

        Returns
        -------
        PlacementResult

        Raises
        ------
        FileNotFoundError
            A FOV's reference image does not exist (the error names the
            setting, ``reference_image`` or ``reference_image_import``, and the
            path).
        ValueError
            An unknown input or positions key, a stage file source without
            ``stage_file``, a FOV's ``PositionID`` that the stage file does not
            hold, a table whose FOVs are not fov_ids, or a reference image without stored metadata while
            ``voxel_size_xy`` or ``voxel_size_z`` is None (naming them).
        IncompatibleGeometryError
            Tile sizes or spacings taken from the grids differ between FOVs, an
            input image of another shape or metadata than its grid, or the
            errors of ``stitch``.
        """
        from starfinder.stitching import ReviewPolicy, TilePositionTable, save_placement, stitch
        from starfinder.dataset._stitching import read_inputs, table_from_positions

        ids = list(fov_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("fov_ids must be a nonempty list of unique FOV IDs")
        from starfinder.dataset._declared import reference_settings
        loaded = read_inputs(self.output_root, ids, input, reference=reference_settings(self))
        grids = {fov: item.grid for fov, item in loaded.items()}
        if not isinstance(table, TilePositionTable):
            table = table_from_positions(table, ids, grids, fov_pattern=self.fov_pattern, position_ids=position_ids)
        if set(table.fov_ids) != set(ids):
            missing = [f for f in ids if f not in table.fov_ids]
            extra = [f for f in table.fov_ids if f not in ids]
            raise ValueError(f"the tile position table must hold exactly fov_ids; missing {missing}, extra {extra}")
        result = stitch([loaded[fov] for fov in table.fov_ids], table, config=config,
                        review=ReviewPolicy() if review is None else review, fallback=fallback, sample=sample)
        if output_dir is not None:
            save_placement(result, output_dir)
        return result

    def export(self, sample: str, *, placement, assignment: str | None = "default", config=None, output_dir):
        """Write a sample's export folder: the sample tables, ``sample.h5ad``, the fused image and, on request,
        the label image and the SpatialData store.

        The one sample-level export entry (design/specs/sample-output.md, "The
        export entry"). A sample-level computation, not a multi-FOV runner: it
        reads only saved files. After the checks before anything is written
        (the configs; the placement's review; the ``anndata`` extra, and
        ``checkpoint`` for Parquet tables; the assembly; the fusion inputs; the
        label IDs; the route, check 7 of design/specs/sample-segmentation.md:
        ``config.route``, or by default ``after_fusion`` for
        ``dimensionality: projection`` and ``per_fov`` for ``stack`` and
        ``both``) it removes the files the previous ``export.json`` lists for
        each part that runs or is off, then calls the parts in order, writing
        ``export.json`` after each: ``assemble`` on one
        :class:`~starfinder.assembly.AssemblyInput` per FOV of the placement
        (``fovs/{fovID}/assignment/<assignment>/`` and the grid of the
        reference image ``reference_image_import`` or ``reference_image``
        names, default ``fovs/{fovID}/reference.tif``: its stored metadata, or for a file
        that stores none the grid ``voxel_size_xy`` and ``voxel_size_z``
        declare), saved with
        :func:`~starfinder.assembly.save_assembly` (with ``assemble`` off the
        saved assembly is ``reused``, or the part is ``off`` when there is none);
        :func:`~starfinder.export.fuse_images` on one
        :class:`~starfinder.export.FusionInput` per included FOV
        (``reference_stain``:
        ``fovs/{fovID}/other_rounds/reference_stain/image.ome.tif``;
        ``reference_merged``: the FOV's reference image;
        ``<name>=file:<pattern>`` relative to the output root); and, with
        ``fuse_labels`` on (off by default),
        :func:`~starfinder.export.fuse_labels` on the placement and the
        assembly, which writes ``cell_labels.ome.tif``; and, with
        ``export_spatialdata`` on (off by default),
        :func:`~starfinder.export.export_spatialdata` on the saved
        assembly and the ``fusion`` and ``labels`` sections, which writes
        ``sample.zarr/`` (with it on, a sample name outside spatialdata's
        element naming rule is refused among the config checks). A part that
        cannot run is recorded ``not_run`` with
        :class:`~starfinder.export.ExportIncompleteWarning`; any other
        error inside the SpatialData part records it ``failed`` and is raised
        again. The part ``segment`` of the route after fusion is recorded
        ``off`` on the route per FOV (reason ``route_per_fov``, or ``no_cells``
        with ``assignment=None``). On the route after fusion with an
        assignment (design/specs/sample-segmentation.md) the parts run in the
        order ``fuse``; ``segment``
        (:func:`~starfinder.export.segment_fused` on the fused
        projection, with the dataset's ``(dataset_id, sample_id)``; with
        ``segment`` off the saved sample segmentation is ``reused`` under the
        reuse rule, checked before anything is written); ``assemble`` on each
        FOV's molecule table ``fovs/{fovID}/molecules.csv`` with the sample
        segmentation (the ``molecules_only`` assembly when ``segment`` could
        not run); then ``fuse_labels`` (the sample's territories as its one
        view) and ``export_spatialdata``, as on the route per FOV.

        Parameters
        ----------
        sample : str
            The sample; the placement must be of this sample.
        placement : PlacementResult, str or Path
            The placement, or its saved folder ``stitching/<sample>/``.
        assignment : str or None
            The assignment name read per FOV, or None for a sample without
            segmentation: each FOV's ``fovs/{fovID}/molecules.csv``
            (``FOV.export_spots``) is read with
            :func:`~starfinder.assignment.load_molecules`, its identities kept,
            and the dataset's gene list.
        config : ExportConfig or None
            None is ``ExportConfig()``.
        output_dir : str or Path
            The sample's export folder; the workflow passes
            ``<OUTPUT>/export/<sample>/``.

        Returns
        -------
        ExportRecord

        Raises
        ------
        ValueError, PlacementReviewRequiredError, ModuleNotFoundError, IncompatibleGeometryError
            The checks of design/specs/sample-output.md ("Checks before
            anything is written"), in their order, with the folder unchanged;
            a reference image without stored metadata while ``voxel_size_xy``
            or ``voxel_size_z`` is None raises ``ValueError`` naming them.
        starfinder.segmentation.MissingModelError, starfinder.segmentation.ModelHashMismatchError
            Check 7: a model of ``config.segmentation`` is missing or has
            another SHA-256, with the folder unchanged.
        """
        # §2.9
        from starfinder.export import ExportConfig
        from starfinder.dataset._export import export_sample
        return export_sample(self, sample, placement=placement, assignment=assignment,
                             config=ExportConfig() if config is None else config, output_dir=output_dir)

    def report(self, sample: str, fov_ids, *, assignment: str | None = "default", segmentation: str | None = None,
               config=None, output_dir=None):
        """Write a sample's processing report: ``report.html``, ``report.json`` and ``tables/``.

        The one sample report entry (docs/report-contract.md, "The sample
        entry"). A sample-level reader, not a runner: it calls no processing
        stage, reads only the saved records at their fixed places under the
        output root ("Where each record is looked for": the FOVs'
        ``fovs/{fovID}/run.json`` and saved transforms, segmentation and
        assignment records, the reference images, ``stitching/<sample>/`` and
        ``export/<sample>/``) and the raw input of the main example for its
        replay, and writes only its output folder. The report describes the
        run and judges nothing. After the checks before anything is written
        (the arguments; a run record or an export to report; each run record's
        IDs; the export and the placement against the FOV list; the marker
        genes against the gene list; the example FOVs; the output folder) the
        report is written into a staging folder that replaces the earlier
        report of the sample once it is complete. A part whose input is
        missing or changed is unavailable with its reason, recorded in
        ``tables/unavailable.csv``; the entry does not raise for it. Run
        records of versions 1 and 2 are read; the relative input paths of a
        version-2 record resolve against this dataset's ``input_root``, so a
        copied output and input tree reports at its new place.

        Parameters
        ----------
        sample : str
            The sample, as ``stitch`` and ``export`` name it.
        fov_ids : Sequence[str]
            The sample's FOVs, nonempty and unique, in the order the report
            shows them.
        assignment : str or None
            The assignment name of the route per FOV, read at
            ``fovs/{fovID}/assignment/<assignment>/``; None for a sample
            without cells.
        segmentation : str or None
            The segmentation run whose outlines the examples show; None is the
            run the assignment (route per FOV) or the export (route after
            fusion) read.
        config : ReportConfig or None
            None is ``ReportConfig()``.
        output_dir : str or Path or None
            None is ``<output_root>/report/<sample>/``.

        Returns
        -------
        ReportRecord
            The folder and the content of its ``report.json``.

        Raises
        ------
        TypeError
            config is not a ReportConfig.
        ValueError
            Arguments of the wrong form, a run record, export or placement
            that does not belong to the sample or the FOV list, a marker gene
            not in the gene list, or example FOVs outside the list.
        FileNotFoundError
            No FOV of the list has a run record and there is no export.
        FileExistsError
            The output folder holds a file that is not one of the report's own.
        """
        # §2.9
        from starfinder.report._collect import report_sample
        return report_sample(self, sample, fov_ids, assignment=assignment, segmentation=segmentation, config=config,
                             output_dir=output_dir)

    def compare_samples(self, samples, *, report_root=None):
        """Write the dataset's comparison page ``index.html`` and ``samples.csv`` from finished sample reports.

        The comparison entry (docs/report-contract.md, "The comparison
        entry"). It reads only the sample reports: each sample's
        ``<report_root>/<sample>/report.json`` and the tables it lists, each
        checked against its recorded SHA-256, and draws one row per sample and
        three box plots by sample (spots per FOV, fraction matched to the
        codebook, cells per FOV) from them. A sample whose folder or
        ``report.json`` is missing is a row marked ``missing``. It reads no run
        record, placement or export, recomputes nothing a sample report holds
        and writes nothing under a sample's folder; ``index.html`` and
        ``samples.csv`` replace earlier ones.

        Parameters
        ----------
        samples : Sequence[str]
            The samples, nonempty and unique, in the order of the page.
        report_root : str or Path or None
            None is ``<output_root>/report/``, where ``Dataset.report`` writes
            each sample's report.

        Returns
        -------
        ComparisonRecord
            The report root and the record of what was written.

        Raises
        ------
        ValueError
            samples is empty or repeats a sample, or a sample's report.json or
            one of the files it lists differs from its record (naming the
            file); nothing is written then.
        """
        from starfinder import _layout
        from starfinder.report._compare import compare_samples
        root = _layout.report_root(self.output_root) if report_root is None else report_root
        return compare_samples(root, samples, dataset_id=self.dataset_id)
