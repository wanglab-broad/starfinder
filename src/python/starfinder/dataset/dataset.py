"""STARMapDataset: sample-level configuration and FOV factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from starfinder.barcode import Codebook, EncodingConfig, load_codebook
from starfinder.dataset.types import (
    ChannelOrder,
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

    Parameters
    ----------
    input_root : Path
        Resolved input sample directory (Path), before round/FOV components.
    output_root : Path
        Resolved output directory (Path).
    dataset_id : str
        Dataset identifier.
    sample_id : str
        Input sample identifier.
    output_id : str
        Output run identifier.
    layers : LayerState
        Shared round categories/reference; default new empty LayerState.
    channel_order : ChannelOrder
        Ordered channel filename patterns; default empty list, set before loading.
    codebook : Codebook | None
        Shared loaded Codebook, default None.
    subtile : SubtileConfig | None
        Shared SubtileConfig, default None.
    rotate_angle : float
        Stored angle in degrees, default 0; call rotate or pass streaming rotate_angle explicitly.
    maximum_projection : bool
        Default False; save_ref_merged projects along Z when True.
    fov_pattern : str
        Percent-format FOV naming pattern; default Position%03d.

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
        """Create dataset from validated Snakemake config dict.

        Handles both direct Python API keys and Snakemake config keys:

        - ``channel_order`` or ``seq_channel_order`` → channel_order

        - ``fov_id_pattern`` or ``fov_pattern`` → fov_pattern

        Parameters
        ----------
        config : dict
            Required: n_rounds, ref_round, root_input_path, root_output_path,
            dataset_id, sample_id, output_id. Channel and FOV aliases are described
            above. Optional rotate_angle, maximum_projection and subtile settings
            are read from config; this factory does not run schema validation.

        Returns
        -------
        STARMapDataset
            Dataset with configured LayerState; does not load images or codebook.
            Call layers.validate() explicitly to check round invariants.

        Raises
        ------
        KeyError
            Required keys are missing.
        """
        layers = LayerState(
            seq=[f"round{i}" for i in range(1, config["n_rounds"] + 1)],
            ref=config["ref_round"],
        )
        # Snakemake config uses seq_channel_order; Python API uses channel_order
        channel_order = config.get("channel_order") or config.get(
            "seq_channel_order", []
        )
        fov_pattern = config.get("fov_id_pattern", config.get("fov_pattern", "Position%03d"))

        sdata = cls(
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
            channel_order=channel_order,
            rotate_angle=config.get("rotate_angle", 0.0),
            maximum_projection=config.get("maximum_projection", False),
            fov_pattern=fov_pattern,
        )

        # Set up subtile config if applicable
        rule_params = config.get("rules", {})
        for rule_name in ("gr_single_fov_subtile", "deep_create_subtile"):
            subtile_params = (
                rule_params.get(rule_name, {})
                .get("parameters", {})
                .get("create_subtiles", {})
            )
            if subtile_params.get("run") or subtile_params.get("sqrt_pieces"):
                sqrt_pieces = subtile_params.get("sqrt_pieces", 4)
                subtile = SubtileConfig(sqrt_pieces=sqrt_pieces)
                subtile.compute_windows(
                    height=config.get("img_row", 0),
                    width=config.get("img_col", 0),
                )
                sdata.subtile = subtile
                break

        return sdata

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
        self.codebook = load_codebook(path, round_labels=tuple(self.layers.seq),
            channel_labels=tuple(self.channel_order),
            encoding=EncodingConfig(reverse_bases=reverse_bases, split_index=split_index))
