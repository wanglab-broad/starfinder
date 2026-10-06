"""The segmentation method registry: exact config type -> SegmentationSpec, and the private method functions.

Each config's method discriminator equals its SEGMENTATION_METHODS spec name
(docs/segmentation-contract.md, "Method registry"; docs/method-registry.md).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from starfinder._registry import Dependency, check_shared

from ._labels import TARGETS, ReferenceGrid
from ._learned import CellposeConfig, StarDistConfig, _cellpose, _stardist

ROLES = ("nuclear", "cytoplasm", "membrane", "amplicon", "composite")
SEEDS = ("none", "optional", "required")
DEVICES = ("cpu", "cuda")


@dataclass(frozen=True)
class MethodContext:
    """What segment passes to a method function besides the image and config.

    roles are the channel roles in channel order; seeds is the seed run's
    ``uint32`` label array on the input grid, or None; grid is the input's
    ReferenceGrid; device is the checked device; model is the resolved model
    folder or file (None for a method without a model).
    """
    roles: tuple[str, ...]
    seeds: np.ndarray | None
    grid: ReferenceGrid
    device: str = "cpu"
    model: Path | None = None


def _subset(value, allowed, name, spec_name):
    if not isinstance(value, frozenset) or not value or not value <= set(allowed):
        raise ValueError(f"{name} of {spec_name!r} must be a nonempty frozenset of {', '.join(map(str, allowed))}")


@dataclass(frozen=True)
class SegmentationSpec:
    """Registered segmentation method: stable snake_case name, private method function and declared capabilities.

    run(image, config, context) receives the validated ZYXC image, the config
    and a :class:`MethodContext`, and returns an integer label array of the
    image's ZYX shape (a method may drop the Z axis of a Z=1 input; the wrapper
    restores it) and a details mapping (``effective`` parameters,
    ``library_dtype``, counts). segment calls it after the stage checks;
    callers never call it directly. targets is the subset of ``nucleus`` and
    ``cell`` the method produces; roles the channel roles it accepts;
    required_roles one frozenset per requirement, of which one role must be
    present; seeds ``none``, ``optional`` or ``required``; dimensions holds 2
    when a Z=1 input runs as a plane and 3 when Z>1 runs as a volume; models is
    True when the config names a model (``model`` or ``model_path``); devices
    the subset of ``cpu`` and ``cuda`` it runs on. requires lists optional
    dependencies, imported when the method runs; min_shape_zyx is the smallest
    accepted size of each axis (a Z=1 input is checked against its last two
    entries).
    """

    name: str
    run: Callable[..., tuple[np.ndarray, dict]]
    _: KW_ONLY
    requires: tuple[Dependency, ...] = ()
    min_shape_zyx: tuple[int, int, int] = (1, 1, 1)
    targets: frozenset[str]
    roles: frozenset[str]
    required_roles: tuple[frozenset[str], ...]
    seeds: str
    dimensions: frozenset[int]
    models: bool
    devices: frozenset[str]

    def __post_init__(self):
        check_shared(self, "segmentation method")
        _subset(self.targets, TARGETS, "targets", self.name)
        _subset(self.roles, ROLES, "roles", self.name)
        if (not isinstance(self.required_roles, tuple)
                or any(not isinstance(r, frozenset) or not r or not r <= self.roles for r in self.required_roles)):
            raise ValueError(f"required_roles of {self.name!r} must be a tuple of nonempty frozensets of its roles")
        if self.seeds not in SEEDS:
            raise ValueError(f"seeds of {self.name!r} must be none, optional or required")
        _subset(self.dimensions, (2, 3), "dimensions", self.name)
        if not isinstance(self.models, bool):
            raise TypeError(f"models of {self.name!r} must be Boolean")
        _subset(self.devices, DEVICES, "devices", self.name)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


@dataclass(frozen=True)
class SeededWatershedConfig:
    """The nucleus-seeded, stain-guided watershed (docs/segmentation-algorithms.md, "seeded_watershed").

    sigma_um is the standard deviation of the Gaussian smoothing of the stain
    in µm, converted per axis with the input's spacing (an axis of length 1 is
    not smoothed; 1.5 µm is provisional, from the W-306 prototype). threshold
    is ``"otsu"`` (Otsu's threshold of the smoothed stain) or a number on the
    ``img_as_float`` scale ([0, 1] for unsigned integers). compactness and
    connectivity are passed to ``skimage.segmentation.watershed``.
    """

    sigma_um: float = 1.5
    threshold: str | float = "otsu"
    compactness: float = 0.0
    connectivity: int = 1
    method: str = field(default="seeded_watershed", init=False)

    def __post_init__(self):
        _positive(self.sigma_um, "sigma_um")
        if self.threshold != "otsu" and (isinstance(self.threshold, bool) or not isinstance(self.threshold, Real)
                                         or not np.isfinite(self.threshold)):
            raise ValueError("threshold must be 'otsu' or a finite number on the img_as_float scale")
        if (isinstance(self.compactness, bool) or not isinstance(self.compactness, Real)
                or not np.isfinite(self.compactness) or self.compactness < 0):
            raise ValueError("compactness must be a nonnegative finite number")
        if self.connectivity not in (1, 2, 3) or isinstance(self.connectivity, bool):
            raise ValueError("connectivity must be 1, 2 or 3")


def _seeded_watershed(image, config: SeededWatershedConfig,
                      context: MethodContext) -> tuple[np.ndarray, dict[str, Any]]:
    """Grow one cell from each seed over the smoothed stain (the W-306 prototype, scripts/seeded_watershed.py).

    smooth = gaussian(img_as_float(stain), sigma_um / spacing per axis, 0 for an
    axis of length 1); foreground = smooth > threshold; the mask is the
    foreground united with every seed voxel, so every seed, also one outside
    the foreground, keeps its value; cells = watershed(-smooth, markers=seeds,
    mask). The stain is the input's one channel.
    """
    from skimage.filters import gaussian, threshold_otsu
    from skimage.segmentation import watershed
    from skimage.util import img_as_float

    if image.shape[-1] != 1:
        raise ValueError(f"seeded_watershed grows cells on one stain channel; the input has roles {context.roles}")
    spacing = context.grid.metadata.spacing_zyx
    if spacing is None:
        raise ValueError("seeded_watershed needs a spacing: the input grid's metadata has no spacing_zyx")
    stain, seeds = image[..., 0], context.seeds
    sigma_px = tuple(0.0 if n == 1 else config.sigma_um / s for n, s in zip(stain.shape, spacing))
    smooth = gaussian(img_as_float(stain), sigma=sigma_px)
    if config.threshold == "otsu":
        value, source = (float(threshold_otsu(smooth)) if smooth.max() > smooth.min() else float(smooth.max()),
                         "otsu")
    else:
        value, source = float(config.threshold), "config"
    foreground = smooth > value
    mask = foreground | (seeds > 0)
    cells = watershed(-smooth, markers=seeds, mask=mask, connectivity=config.connectivity,
                      compactness=config.compactness)
    seed_ids = np.unique(seeds[seeds > 0])
    outside = [int(k) for k in seed_ids if not foreground[seeds == k].any()]
    details = {"effective": {"sigma_px_zyx": list(sigma_px), "threshold": value, "threshold_source": source},
               "library_dtype": str(cells.dtype), "label_rule": "seed_values",
               "counts": {"seeds": int(len(seed_ids)), "seeds_outside_foreground": outside,
                          "foreground_fraction": float(foreground.mean())}}
    return cells, details


#: The segmentation registry (docs/segmentation-contract.md, "Registered methods"). The backends of
#: stardist and cellpose are imported by segment (check 5) and inside the method functions only.
SEGMENTATION_METHODS: dict[type, SegmentationSpec] = {
    StarDistConfig: SegmentationSpec(
        "stardist", _stardist,
        requires=(Dependency("stardist", "stardist", "stardist"), Dependency("csbdeep", "csbdeep", "stardist"),
                  Dependency("tensorflow", "tensorflow", "stardist")),
        targets=frozenset({"nucleus", "cell"}), roles=frozenset({"nuclear", "composite"}),
        required_roles=(frozenset({"nuclear", "composite"}),), seeds="none", dimensions=frozenset({2, 3}),
        models=True, devices=frozenset({"cpu", "cuda"})),
    CellposeConfig: SegmentationSpec(
        "cellpose", _cellpose,
        requires=(Dependency("cellpose", "cellpose", "cellpose"), Dependency("torch", "torch", "cellpose")),
        targets=frozenset({"nucleus", "cell"}), roles=frozenset({"cytoplasm", "nuclear"}),
        required_roles=(frozenset({"cytoplasm", "nuclear"}),), seeds="none", dimensions=frozenset({2, 3}),
        models=True, devices=frozenset({"cpu", "cuda"})),
    SeededWatershedConfig: SegmentationSpec(
        "seeded_watershed", _seeded_watershed, targets=frozenset({"cell"}),
        roles=frozenset({"cytoplasm", "membrane", "amplicon", "composite"}),
        required_roles=(frozenset({"cytoplasm", "membrane", "amplicon", "composite"}),), seeds="required",
        dimensions=frozenset({2, 3}), models=False, devices=frozenset({"cpu"})),
}
