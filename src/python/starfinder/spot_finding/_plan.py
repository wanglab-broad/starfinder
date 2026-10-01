"""The detection plan: one method for every channel, with whole-config overrides for named channels."""
from dataclasses import dataclass

from starfinder._registry import spec_for

from ._methods import SPOT_FINDING_METHODS, SpotFindingConfig


def _check_columns(spec, plan_config, override):
    """Raise ValueError when an override changes a field that adds an optional output column."""
    for column, name in spec.column_fields.items():
        if getattr(override.config, name) != getattr(plan_config, name):
            raise ValueError(f"the override of channel {override.channel!r} sets {name}="
                             f"{getattr(override.config, name)!r}, unlike the plan's config, which would change the "
                             f"output column {column!r}; an override may change any setting except one that "
                             "changes the output columns")


def _check_config(config):
    spec = spec_for(SPOT_FINDING_METHODS, config, "spot-finding method", TypeError, "unsupported detection config")
    config.__post_init__()
    return spec


@dataclass(frozen=True)
class ChannelOverride:
    """The whole config of one channel, named by its channel label.

    config must have the exact type of the plan's config (one method per
    run); its channel_labels must be None or equal to the plan's. It may
    change any setting except one that changes the output columns (the
    fields in the method's column_fields, such as measure_peak_intensity).
    """
    channel: str
    config: SpotFindingConfig

    def __post_init__(self):
        if not isinstance(self.channel, str) or not self.channel:
            raise ValueError("a channel override names a nonempty channel label")
        _check_config(self.config)


@dataclass(frozen=True)
class SpotFindingPlan:
    """The method and its settings for every channel, and the channels whose whole config is replaced.

    A bare config means SpotFindingPlan(config). Override channels are
    unique; when config.channel_labels is set they must be among them
    (FOV.find_spots and FOV.run fill the labels from Dataset.channel_order).
    Direct find_spots with overrides needs config.channel_labels.
    rounds names the rounds to detect in: None (the default) is the
    reference round only, with every result unchanged; a tuple of unique
    round labels (labels of RoundState.all_rounds, checked by FOV.find_spots
    and FOV.run) gives one table with a ``round`` column. Direct find_spots
    detects one image and accepts only rounds=None.
    """
    config: SpotFindingConfig
    channel_overrides: tuple[ChannelOverride, ...] = ()
    rounds: tuple[str, ...] | None = None

    def __post_init__(self):
        spec = _check_config(self.config)
        if not isinstance(self.channel_overrides, (tuple, list)):
            raise TypeError("channel_overrides must be a tuple of ChannelOverride")
        overrides = tuple(self.channel_overrides)
        object.__setattr__(self, "channel_overrides", overrides)
        if self.rounds is not None:
            if not isinstance(self.rounds, (tuple, list)):
                raise TypeError("rounds must be None or a tuple of round labels")
            rounds = tuple(self.rounds)
            if not rounds or any(not isinstance(r, str) or not r for r in rounds) or len(set(rounds)) != len(rounds):
                raise ValueError(f"rounds must be None or nonempty unique round labels, not {rounds!r}")
            object.__setattr__(self, "rounds", rounds)
        labels = self.config.channel_labels
        seen = set()
        for override in overrides:
            if not isinstance(override, ChannelOverride):
                raise TypeError("channel_overrides must be a tuple of ChannelOverride")
            override.__post_init__()
            if type(override.config) is not type(self.config):
                raise TypeError(f"the override of channel {override.channel!r} is a "
                                f"{type(override.config).__name__}, not the plan's {type(self.config).__name__}")
            _check_columns(spec, self.config, override)
            if override.config.channel_labels not in (None, labels):
                raise ValueError(f"the override of channel {override.channel!r} has other channel_labels "
                                 "than the plan's config")
            if override.channel in seen:
                raise ValueError(f"channel {override.channel!r} is overridden more than once")
            seen.add(override.channel)
            if labels is not None and override.channel not in labels:
                raise ValueError(f"channel {override.channel!r} is not one of the channel labels {list(labels)}")
