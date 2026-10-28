"""The stitching method registry: exact config type -> StitchingSpec, and what a method returns.

design/specs/stitching.md ("Method registry"), on the shared mechanism of
docs/method-registry.md. No method is registered here; each method's module adds its
entry (``_multiview``: ``multiview_stitcher``). ``supplied`` (:func:`placement_from_table`)
and ``bigstitcher`` (the import) are not methods and are never keys.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from starfinder._registry import Dependency, check_shared

WHAT = "stitching method"
METHOD_LINK_COLUMNS = ("fov_a", "fov_b", "shift_z", "shift_y", "shift_x", "quality", "kept", "reason")
# Absolute, per matrix entry: how far a linear part may be from the identity and still be a
# translation, for the BigStitcher import and the multiview_stitcher adapter alike.
LINEAR_TOLERANCE = 1e-9


@dataclass(frozen=True)
class StitchingSpec:
    """Registered stitching method: stable snake_case name, private method function and declared capabilities.

    ``run(tiles, grids, nominal_zyx, pairs, *, config)`` receives the tiles (ZYX
    arrays in table order, ``None`` for an absent image, which the method must
    leave unplaced), their reference grids, the ``(n, 3)`` nominal origins in µm
    and the :class:`NeighborPair` list, and returns a :class:`MethodPlacement`.
    ``stitch`` calls it once after its checks; callers never call it directly.
    ``requires`` lists optional dependencies, imported when the method runs;
    ``min_shape_zyx`` is the smallest accepted size of each axis (a Z=1 grid is
    checked against its last two entries); ``dimensions`` holds 2 when Z=1 tiles
    are placed as planes and 3 when Z>1 tiles are placed as volumes; ``devices``
    is ``("cpu",)`` (placement is CPU only) and ``transform_kind`` is
    ``"translation"``, the only kind the stitching stage admits.

    Raises
    ------
    ValueError, TypeError
        A shared field outside the rules of docs/method-registry.md, dimensions
        that are not a nonempty subset of {2, 3}, devices other than
        ``("cpu",)`` or another transform kind.
    """
    # §2.10

    name: str
    run: Callable
    requires: tuple[Dependency, ...] = ()
    min_shape_zyx: tuple[int, int, int] = (1, 1, 1)
    dimensions: frozenset[int] = frozenset({2, 3})
    devices: tuple[str, ...] = ("cpu",)
    transform_kind: str = "translation"

    def __post_init__(self):
        check_shared(self, WHAT)
        if not isinstance(self.dimensions, frozenset) or not self.dimensions or not self.dimensions <= {2, 3}:
            raise ValueError(f"dimensions of {self.name!r} must be a nonempty frozenset of 2 and 3")
        if self.devices != ("cpu",):
            raise ValueError(f"devices of {self.name!r} must be ('cpu',): placement runs on the CPU only")
        if self.transform_kind != "translation":
            raise ValueError(f"transform_kind of {self.name!r} must be 'translation', the only kind §2.10 admits")


@dataclass(frozen=True, eq=False)
class MethodPlacement:
    """What a stitching method returns in its one call.

    ``translations_zyx`` is ``(n, 3)`` float64 in µm, view to common frame, in
    table order: the FOV's correction of its nominal origin before the gauge
    (NaN where not placed). ``components`` is ``(n,)`` int, the method's own
    component labels (recorded as a diagnostic; ``stitch`` derives the components
    from the kept links). ``links`` has the columns ``fov_a``, ``fov_b``,
    ``shift_z/y/x`` (µm, of ``fov_b`` against ``fov_a``), ``quality``, ``kept``
    and ``reason`` (why an unkept link was not kept), one row per registered
    neighbour pair. ``diagnostics`` holds library-specific values (removed links,
    iterations, reference view); its JSON values are recorded.
    """

    translations_zyx: np.ndarray
    components: np.ndarray
    links: pd.DataFrame
    diagnostics: dict = field(default_factory=dict)


#: The stitching method registry, mapping each exact frozen config type to its StitchingSpec.
STITCHING_METHODS: dict[type, StitchingSpec] = {}
