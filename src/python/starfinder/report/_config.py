"""The configuration of the processing report (docs/report-contract.md, "Configuration")."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


def _is_int(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _name(value) -> bool:
    return isinstance(value, str) and bool(value)


@dataclass(frozen=True)
class MarkerGroup:
    """One figure of the marker patterns: a title and one to four distinct gene names.

    Raises
    ------
    TypeError
        name is not a string or genes not a sequence of strings.
    ValueError
        An empty name, no gene or more than four, an empty or repeated gene name.
    """

    name: str
    genes: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError(f"MarkerGroup.name must be a string; got {self.name!r}")
        if not self.name:
            raise ValueError("MarkerGroup.name must be a nonempty string")
        if isinstance(self.genes, str) or not isinstance(self.genes, (tuple, list)):
            raise TypeError(f"MarkerGroup.genes must be a sequence of gene names; got {self.genes!r}")
        genes = tuple(self.genes)
        if not all(isinstance(g, str) for g in genes):
            raise TypeError(f"MarkerGroup.genes must be strings; got {genes!r}")
        if not 1 <= len(genes) <= 4:
            raise ValueError(f"marker group {self.name!r} must hold 1 to 4 genes; got {len(genes)}")
        if not all(genes) or len(set(genes)) != len(genes):
            raise ValueError(f"marker group {self.name!r} must hold distinct nonempty gene names; got {list(genes)}")
        object.__setattr__(self, "genes", genes)


@dataclass(frozen=True)
class RuleLimit:
    """The limit of one problematic-FOV rule.

    ``absolute`` set: that limit alone. ``k`` and ``floor`` set: a deviation from
    the sample median larger than ``max(k × 1.4826 × MAD, floor)``. All three None:
    the rule is not evaluated. Each given value is a finite number ≥ 0.

    Raises
    ------
    TypeError
        A value that is not a number or None.
    ValueError
        A negative or non-finite value.
    """

    k: float | None = None
    floor: float | None = None
    absolute: float | None = None

    def __post_init__(self):
        for name in ("k", "floor", "absolute"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"RuleLimit.{name} must be a number or None; got {value!r}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"RuleLimit.{name} must be a finite number ≥ 0; got {value!r}")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class ReportLimits:
    """The limits of the problematic-FOV rules; the defaults are the provisional limits of choice 14.

    Raises
    ------
    TypeError
        A rule's limit that is not a RuleLimit, or spots_near_zero not a number.
    ValueError
        spots_near_zero outside (0, 1).
    """

    shift_residual: RuleLimit = RuleLimit(k=5.0, floor=5.0)
    alignment_score: RuleLimit = RuleLimit()
    correct_form_fraction: RuleLimit = RuleLimit(k=3.0, floor=0.05)
    matched_fraction: RuleLimit = RuleLimit(k=3.0, floor=0.05)
    spots_near_zero: float = 0.05

    def __post_init__(self):
        for name in ("shift_residual", "alignment_score", "correct_form_fraction", "matched_fraction"):
            if not isinstance(getattr(self, name), RuleLimit):
                raise TypeError(f"ReportLimits.{name} must be a RuleLimit; got {getattr(self, name)!r}")
        value = self.spots_near_zero
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"ReportLimits.spots_near_zero must be a number; got {value!r}")
        if not 0 < value < 1:
            raise ValueError(f"ReportLimits.spots_near_zero must be in (0, 1); got {value!r}")
        object.__setattr__(self, "spots_near_zero", float(value))


@dataclass(frozen=True)
class ReportConfig:
    """What the user supplies to the sample report.

    ``markers``: groups of one to four genes, each one figure, names unique;
    empty shows the three most abundant genes. ``marker_pairs``: pairs of two
    distinct genes for MECR; empty makes MECR unavailable (``no_markers``).
    ``positive_threshold``: the whole count (an integer ≥ 1) at which a cell is
    positive for a gene. ``example_fovs``: three distinct FOVs, or None for the
    automatic choice. ``limits``: the problematic-FOV limits. ``figure_width_px``
    (a positive integer) and ``jpeg_quality`` (1 to 95): the whole-sample figures.
    Lists are kept as tuples.

    Raises
    ------
    TypeError
        A field of the wrong type.
    ValueError
        A value outside its range, repeated marker group names, a pair that is
        not two distinct gene names, example FOVs that are not three distinct
        names.
    """
    # D4

    markers: tuple[MarkerGroup, ...] = ()
    marker_pairs: tuple[tuple[str, str], ...] = ()
    positive_threshold: int = 1
    example_fovs: tuple[str, str, str] | None = None
    limits: ReportLimits = ReportLimits()
    figure_width_px: int = 1600
    jpeg_quality: int = 85

    def __post_init__(self):
        if isinstance(self.markers, str) or not isinstance(self.markers, (tuple, list)):
            raise TypeError(f"ReportConfig.markers must be a sequence of MarkerGroup; got {self.markers!r}")
        markers = tuple(self.markers)
        if not all(isinstance(m, MarkerGroup) for m in markers):
            raise TypeError("ReportConfig.markers must hold MarkerGroup entries")
        names = [m.name for m in markers]
        if len(set(names)) != len(names):
            raise ValueError(f"ReportConfig.markers must have unique names; got {names}")
        if isinstance(self.marker_pairs, str) or not isinstance(self.marker_pairs, (tuple, list)):
            raise TypeError(f"ReportConfig.marker_pairs must be a sequence of gene pairs; got {self.marker_pairs!r}")
        pairs = []
        for pair in self.marker_pairs:
            if (isinstance(pair, str) or not isinstance(pair, (tuple, list))
                    or not all(isinstance(g, str) for g in pair)):
                raise TypeError(f"each marker pair must be two gene names; got {pair!r}")
            pair = tuple(pair)
            if len(pair) != 2 or not all(pair) or pair[0] == pair[1]:
                raise ValueError(f"each marker pair must be two distinct nonempty gene names; got {list(pair)}")
            pairs.append(pair)
        if not _is_int(self.positive_threshold):
            raise TypeError(f"ReportConfig.positive_threshold must be an integer; got {self.positive_threshold!r}")
        if self.positive_threshold < 1:
            raise ValueError(f"ReportConfig.positive_threshold must be ≥ 1; got {self.positive_threshold!r}")
        examples = self.example_fovs
        if examples is not None:
            if isinstance(examples, str) or not isinstance(examples, (tuple, list)):
                raise TypeError(f"ReportConfig.example_fovs must be three FOV IDs or None; got {examples!r}")
            examples = tuple(examples)
            if len(examples) != 3 or not all(_name(f) for f in examples) or len(set(examples)) != 3:
                raise ValueError(f"ReportConfig.example_fovs must be three distinct FOV IDs; got {list(examples)}")
        if not isinstance(self.limits, ReportLimits):
            raise TypeError(f"ReportConfig.limits must be a ReportLimits; got {self.limits!r}")
        for name, low, high in (("figure_width_px", 1, None), ("jpeg_quality", 1, 95)):
            value = getattr(self, name)
            if not _is_int(value):
                raise TypeError(f"ReportConfig.{name} must be an integer; got {value!r}")
            if value < low or (high is not None and value > high):
                raise ValueError(f"ReportConfig.{name} must be {'≥ 1' if high is None else f'in [{low}, {high}]'}; "
                                 f"got {value!r}")
        object.__setattr__(self, "markers", markers)
        object.__setattr__(self, "marker_pairs", tuple(pairs))
        object.__setattr__(self, "positive_threshold", int(self.positive_threshold))
        object.__setattr__(self, "example_fovs", examples)


def marker_genes(config: ReportConfig) -> tuple[str, ...]:
    """Every gene the markers and marker pairs of a configuration name, in first-seen order."""
    return tuple(dict.fromkeys([g for m in config.markers for g in m.genes]
                               + [g for pair in config.marker_pairs for g in pair]))
