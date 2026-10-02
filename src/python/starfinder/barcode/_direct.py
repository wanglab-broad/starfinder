"""Direct readout: a gene per (round, channel) and assignment from each candidate's own round.

docs/readout-contract.md, "Direct readout"; docs/readout-algorithms.md, "Direct assignment".
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from starfinder.spot_finding import SpotFindingResult
from .codebook import _labels
from .extraction import IntensityExtractionResult

_COLUMNS = ("round", "channel", "gene_id")


@dataclass(frozen=True, eq=False)
class DirectPanel:
    """The gene of each (round, channel) of a direct-readout assay.

    table has the columns round, channel (a channel label) and gene_id, all
    nonempty strings. Each (round, channel) appears once and each gene once; a
    repeated one raises ValueError naming it. A (round, channel) may be absent
    (for example a stain channel): candidates there are unmatched. The entry of a
    (round, channel) is "<round>/<channel>". The repr is a one-line size summary.
    """

    table: pd.DataFrame

    def __post_init__(self):
        table = self.table
        if not isinstance(table, pd.DataFrame):
            raise TypeError("DirectPanel table must be a pandas DataFrame")
        if sorted(table.columns) != sorted(_COLUMNS) or not table.columns.is_unique:
            raise ValueError(f"direct panel columns must be {', '.join(_COLUMNS)}; got {list(table.columns)}")
        for column in _COLUMNS:
            if any(not isinstance(v, str) or not v.strip() for v in table[column]):
                raise ValueError(f"direct panel column {column!r} must hold nonempty strings")
        table = table[list(_COLUMNS)].astype("string").reset_index(drop=True)
        pairs = table[table.duplicated(["round", "channel"], keep=False)]
        if len(pairs):
            first = pairs.iloc[0]
            raise ValueError(f"(round, channel) ({first['round']!r}, {first['channel']!r}) appears more than "
                             "once in the direct panel")
        genes = table.gene_id[table.gene_id.duplicated()]
        if len(genes):
            raise ValueError(f"gene {sorted(set(genes))[0]!r} appears more than once in the direct panel; "
                             "each gene has one (round, channel)")
        object.__setattr__(self, "table", table)

    @property
    def gene_of(self) -> dict[tuple[str, str], str]:
        """The gene of each mapped (round, channel)."""
        return {(r, c): g for r, c, g in zip(self.table["round"], self.table.channel, self.table.gene_id)}

    @property
    def genes(self) -> tuple[str, ...]:
        """The panel's genes in table order."""
        return tuple(self.table.gene_id)

    @property
    def n_genes(self) -> int:
        return len(self.table)

    @property
    def round_labels(self) -> tuple[str, ...]:
        """The rounds the panel names, in first-appearance order."""
        return tuple(dict.fromkeys(self.table["round"]))

    @property
    def channel_labels(self) -> tuple[str, ...]:
        """The channels the panel names, in first-appearance order."""
        return tuple(dict.fromkeys(self.table.channel))

    def __repr__(self):
        return f"DirectPanel: {self.n_genes} genes over {len(self.round_labels)} rounds"


def load_direct_panel(path: str | Path, *, round_labels: tuple[str, ...],
                      channel_labels: tuple[str, ...]) -> DirectPanel:
    """Read a direct panel CSV with the header round,channel,gene_id (UTF-8 BOM accepted).

    Rounds must be among round_labels (the sequencing rounds) and channels among
    channel_labels. Errors identify the source row; a repeated (round, channel) or
    gene names it and both rows. Blank lines are skipped.
    """
    _labels(round_labels, "round_labels")
    _labels(channel_labels, "channel_labels")
    records, seen = [], {"pair": {}, "gene": {}}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = csv.reader(handle)
        header = [v.strip() for v in next(rows, [])]
        if header != list(_COLUMNS):
            raise ValueError(f"{path}: the direct panel header must be {','.join(_COLUMNS)}; got {','.join(header)}")
        for line, values in enumerate(rows, 2):
            if not values:
                continue
            if len(values) != 3:
                raise ValueError(f"{path}: row {line}: expected round,channel,gene_id")
            round_name, channel, gene = (v.strip() for v in values)
            if round_name not in round_labels:
                raise ValueError(f"{path}: row {line}: round {round_name!r} is not a sequencing round "
                                 f"({', '.join(round_labels)})")
            if channel not in channel_labels:
                raise ValueError(f"{path}: row {line}: channel {channel!r} is not a channel label "
                                 f"({', '.join(channel_labels)})")
            if not gene:
                raise ValueError(f"{path}: row {line}: gene_id is empty")
            for kind, key, name in (("pair", (round_name, channel), f"(round, channel) ({round_name!r}, {channel!r})"),
                                    ("gene", gene, f"gene {gene!r}")):
                if key in seen[kind]:
                    raise ValueError(f"{path}: rows {seen[kind][key]} and {line}: {name} appears more than once")
                seen[kind][key] = line
            records.append((round_name, channel, gene))
    return DirectPanel(pd.DataFrame(records, columns=list(_COLUMNS), dtype=object))


@dataclass(frozen=True)
class DirectAssignmentConfig:
    """Direct readout: the gene of each candidate's own round and channel.

    No parameters besides the panel. No codeword competition, rescue, required
    rounds, encoding or end-base check applies, and the brightest channel never
    changes the identity.
    """

    method: str = field(default="direct", init=False)

    def __post_init__(self):
        pass


def _run_direct(intensity_result, spots, panel, config):
    """The direct read table of assign_direct (docs/readout-algorithms.md, "Direct assignment").

    Status precedence: unmapped_channel, then invalid_measurement, then
    zero_signal_round. Negative sums count as 0, as in the decoders' clipping.
    """
    if not isinstance(spots, SpotFindingResult):
        raise TypeError("spots must be the SpotFindingResult of the extracted candidates")
    spots.__post_init__()
    frame = spots.spots
    if "round" not in frame or "channel" not in frame:
        raise ValueError("direct readout needs candidates with round and channel columns: detect with a "
                         "SpotFindingPlan with explicit rounds")
    if (spots.spot_namespace != intensity_result.spot_namespace
            or tuple(frame.spot_id) != intensity_result.spot_ids):
        raise ValueError("spot and intensity identities must match in order")
    rounds, channels = intensity_result.round_labels, intensity_result.channel_labels
    outside = [r for r in panel.round_labels if r not in rounds] + [c for c in panel.channel_labels if c not in channels]
    if outside:
        raise ValueError(f"direct panel labels {outside} are not rounds or channels of the extraction "
                         f"({', '.join(rounds)}; {', '.join(channels)})")
    channel_index = frame.channel.to_numpy(dtype=np.int64)
    if (channel_index >= len(channels)).any():
        raise ValueError("spot channel outside the labeled channels")
    n = len(frame)
    round_names = frame["round"].astype("string").to_numpy(dtype=object)
    position = {name: i for i, name in enumerate(rounds)}
    own = np.array([position.get(r, -1) for r in round_names], dtype=np.int64)
    extracted = own >= 0
    rows = np.flatnonzero(extracted)
    sums = np.zeros((n, len(channels)))
    sums[rows] = np.maximum(intensity_result.values[rows, :, own[rows]], 0)
    valid = np.zeros(n, dtype=bool)
    valid[rows] = intensity_result.valid[rows, own[rows]]
    own_sum = sums[np.arange(n), channel_index]
    total = sums.sum(axis=1)
    measured = valid & (total > 0)
    rank = np.where(measured, 1 + (sums > own_sum[:, None]).sum(axis=1), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(measured, own_sum / total, np.nan)
    labels = [channels[c] for c in channel_index]
    gene_of = panel.gene_of
    genes = np.array([gene_of.get((r, c)) for r, c in zip(round_names, labels)], dtype=object)
    status = np.full(n, "assigned", dtype=object)
    reason = np.full(n, "", dtype=object)
    for mask, value, why in ((valid & (total == 0), "no_signal", "zero_signal_round"),
                             (~valid, "unmatched", "invalid_measurement"),
                             (pd.isna(genes), "unmatched", "unmapped_channel")):
        status[mask], reason[mask] = value, why
    assigned = status == "assigned"
    missing = pd.array([pd.NA] * n, dtype="string")
    table = pd.DataFrame({
        "spot_id": pd.array(intensity_result.spot_ids, dtype="string"),
        "spot_namespace": pd.array([intensity_result.spot_namespace] * n, dtype="string"),
        "round": pd.array(round_names, dtype="string"),
        "channel": pd.array(labels, dtype="string"),
        "observed_color_sequence": missing,
        "decoded_color_sequence": missing.copy(),
        "gene_id": pd.array(np.where(assigned, genes, None), dtype="string"),
        "entry_id": pd.array([f"{r}/{c}" if a else None for r, c, a in zip(round_names, labels, assigned)],
                             dtype="string"),
        "call_status": pd.array(status, dtype="string"),
        "failure_reason": pd.array(reason, dtype="string"),
        "call_type": pd.array(np.where(assigned, "direct", "no_call"), dtype="string"),
        "own_channel_rank": rank.astype("float64"),
        "own_channel_fraction": fraction.astype("float64"),
    })
    return table, {"own_round_only": True}
