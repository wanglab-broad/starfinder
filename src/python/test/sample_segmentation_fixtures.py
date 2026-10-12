"""Fixtures of docs/sample-segmentation-validation.md ("Fixtures"), rebuilt in the test.

``fused_pair``: the fixture of W-401's grounding calls (``scripts/grounding_calls.py``, read,
not copied), two FOVs of 3 × 24 × 24 voxels at spacing (1.0, 0.5, 0.5) µm and origins (0, 0,
0) and (0, 0, 8) µm, cut from one scene of 24 × 40 pixels with three disc cells (cell 1
across the seam, cells 2 and 3 inside one FOV each) and smaller disc nuclei; the stains
``dapi`` (nuclei) and ``flamingo`` (cells), each FOV the scene cut at its origin and
repeated over its three planes; the molecules of the grounding calls. ``fused_quad``: four
FOVs of 3 × 40 × 40 voxels at the origins of the page (fractional offsets, a four-FOV
overlap), cut from one scene of 72 × 72 pixels: one cell at the crossing of the four FOVs,
two across a side seam, one cut by the sample's outer border, one wholly inside each FOV,
four molecules per cell (the border cell's second beyond the union grid, within the far
voxel of FOV 4's footprint), each seen by every FOV whose footprint holds it; two cell IDs
are at and above 2^31. A FOV pixel takes the scene pixel nearest its centre, as
``truth_import`` samples the scene on the segmentation grid. ``fused_pair_seams`` (W-429):
the two seam cases of the comparison on ``fused_pair``'s two FOVs, with the molecules
``m_a`` to ``m_g`` (:func:`seam_spot_ids` names their rows). ``scene_plane_small`` (W-429):
the change of ``scene_plane`` applied to ``sample-small-v1``.

:func:`write_scene` writes a scene as a dataset (the saved placement, ``images/<stain>/``
ZYX TIFFs with their axes, ``images/ref_merged/`` with the grid, ``signal/`` goodSpots
tables, the codebook); :func:`fuse_scene` calls ``fuse`` alone on it at a factor ``k``;
:func:`quad_grid_placement` (W-446) saves ``fused_quad``'s placement with grid cells.
:func:`truth_import`, :func:`seeded_runs`, :func:`truth_runs`, :func:`length_configs` and
:func:`route_config` follow the rows of the same names. Nothing is read from a run
directory.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from starfinder.barcode import Codebook
from starfinder.dataset import Dataset
from starfinder.image import ImageMetadata
from starfinder.io import save_volume
from starfinder.sample_output import (ExportConfig, FusionConfig, FusionInput, SampleSegmentationConfig,
                                      SampleSegmentationRun, fuse)
from starfinder.segmentation import (CellposeConfig, LabelImportConfig, ReferenceGrid, SeededWatershedConfig,
                                     StarDistConfig)
from starfinder.stitching import (approve_placement, placement_from_table, positions_from_grid, positions_from_table,
                                  save_placement)

DS, SAMPLE = "data", "s1"
GENES = ("A", "B")
SPACING = (1.0, 0.5, 0.5)
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
STAINS = ("dapi", "flamingo")
IMAGES = tuple(f"{name}=file:images/{name}/{{fov_id}}.tif" for name in STAINS)
BIG = 2 ** 31


@dataclass
class Scene:
    """A hand-built scene: FOV shape and origins, the scene's cell and nucleus labels (0.5 µm pixels from 0) and
    the molecules of each FOV, ``((z, y, x) index on the FOV grid, gene)``."""
    name: str
    shape: tuple
    origins: dict
    cells: np.ndarray
    nuclei: np.ndarray
    molecules: dict
    statuses: dict = field(default_factory=dict)

    @property
    def ids(self):
        return list(self.origins)

    def grid(self):
        return ReferenceGrid(self.shape, ImageMetadata("fov", SPACING, (0.0, 0.0, 0.0), IDENTITY, "µm"), "declared")

    def stain(self, name):
        if name == "dapi":
            return np.where(self.nuclei > 0, 300, 5).astype(np.uint16)
        return np.where(self.cells > 0, 200, 10).astype(np.uint16)

    def cut(self, scene2d, fov):
        """The FOV's ZYX image: at each pixel the scene pixel nearest its centre, repeated over the planes."""
        origin = self.origins[fov]
        index = [scene_index(origin[a] + np.arange(self.shape[a]) * SPACING[a], scene2d.shape[a - 1])
                 for a in (1, 2)]
        tile = np.where((index[0][:, None] >= 0) & (index[1][None, :] >= 0),
                        scene2d[index[0].clip(0)[:, None], index[1].clip(0)[None, :]], 0).astype(scene2d.dtype)
        return np.repeat(tile[np.newaxis], self.shape[0], axis=0)


def scene_index(position_um, size):
    """The scene pixel nearest a position (µm), ``floor(p / 0.5 + 1/2)``, or −1 outside the scene."""
    index = np.floor(np.asarray(position_um, np.float64) / SPACING[1] + 0.5).astype(np.int64)
    return np.where((index >= 0) & (index < size), index, -1)


def _discs(shape, cells, shrink=0):
    labels = np.zeros(shape, np.uint32)
    yy, xx = np.indices(shape)
    for cell, ((cy, cx), radius) in cells.items():
        labels[(yy - cy) ** 2 + (xx - cx) ** 2 <= max(2, radius - shrink) ** 2] = cell
    return labels


# --- fused_pair -----------------------------------------------------------------------------------------------

PAIR_CELLS = {1: ((12, 20), 5), 2: ((6, 6), 3), 3: ((17, 33), 3)}
PAIR_MOLECULES = {
    # FOV1: x 8.5 µm in cell 1 (owned by FOV1); x 3 µm in cell 2; x 11 µm in cell 1 but owned by FOV2.
    "FOV1": [((1, 12, 17), "A"), ((1, 6, 6), "B"), ((1, 12, 22), "A")],
    # FOV2 (origin x 8 µm): x 11 µm in cell 1 (the same molecule as FOV1's third); x 9.5 µm (owned by FOV1);
    # x 16.5 µm in cell 3.
    "FOV2": [((1, 12, 6), "A"), ((1, 12, 3), "B"), ((1, 17, 17), "B")]}


def fused_pair() -> Scene:
    shape = (24, 40)
    return Scene("fused_pair", (3, 24, 24), {"FOV1": (0.0, 0.0, 0.0), "FOV2": (0.0, 0.0, 8.0)},
                 _discs(shape, PAIR_CELLS), _discs(shape, PAIR_CELLS, shrink=2), PAIR_MOLECULES)


# --- fused_quad -----------------------------------------------------------------------------------------------

#: cell → ((centre y, x) in scene pixels, radius in pixels): the crossing (2^31), a cell across the FOV1 | FOV3
#: seam and one across the FOV1 | FOV2 seam, the border cell (2^32 − 1), one inside each FOV.
QUAD_CELLS = {BIG: ((35, 35), 5), 2: ((14, 35), 4), 3: ((35, 14), 4), 2 ** 32 - 1: ((67, 55), 5),
              11: ((8, 8), 4), 12: ((58, 10), 4), 13: ((10, 58), 4), 14: ((54, 54), 4)}
QUAD_ORIGINS = {"FOV1": (0.0, 0.0, 0.0), "FOV2": (0.0, 15.25, 0.25), "FOV3": (0.0, 0.25, 15.25),
                "FOV4": (0.0, 15.25, 15.25)}
#: The four molecules of a cell, offsets (y, x) in µm from its centre; the border cell's second lies at y 34.9 µm,
#: beyond the union grid's last voxel (34.75 µm) and inside FOV4's footprint (35.0 µm).
QUAD_OFFSETS = ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0))
QUAD_BORDER_OFFSETS = ((-1.0, 0.0), (1.4, 0.0), (0.0, -1.0), (0.0, 1.0))


def quad_molecules(shape=(3, 40, 40), origins=QUAD_ORIGINS, cells=QUAD_CELLS):
    """Per FOV ``((z, y, x) index, gene)`` of every cell molecule whose position the FOV's footprint holds."""
    out = {fov: [] for fov in origins}
    for n, (cell, ((cy, cx), _)) in enumerate(cells.items()):
        offsets = QUAD_BORDER_OFFSETS if cell == 2 ** 32 - 1 else QUAD_OFFSETS
        for m, (dy, dx) in enumerate(offsets):
            y, x = cy * SPACING[1] + dy, cx * SPACING[2] + dx
            for fov, origin in origins.items():
                iy, ix = (y - origin[1]) / SPACING[1], (x - origin[2]) / SPACING[2]
                if -0.5 <= iy < shape[1] - 0.5 and -0.5 <= ix < shape[2] - 0.5:
                    out[fov].append(((1.0, iy, ix), GENES[(n + m) % 2]))
    return out


def fused_quad() -> Scene:
    shape = (72, 72)
    return Scene("fused_quad", (3, 40, 40), dict(QUAD_ORIGINS), _discs(shape, QUAD_CELLS),
                 _discs(shape, QUAD_CELLS, shrink=2), quad_molecules())


# --- fused_quad_notch (W-427) -------------------------------------------------------------------------------------

#: The L-shaped cell around the notch that the placeholder FOV4 leaves: an arm down x 17–19.5 µm (scene columns 34 to
#: 39, rows 34 to 66) and an arm along y 17–19.5 µm (rows 34 to 39, columns 40 to 58). Its centroid, about (22.5,
#: 20.5) µm, lies where no owning footprint covers it (FOV2 ends at x 20 µm, FOV3 at y 20 µm); the nearest footprint
#: centre is FOV2's (25, 10) µm.
NOTCH_CELL = BIG + 5
NOTCH_DISCS = {cell: value for cell, value in QUAD_CELLS.items() if cell in (2, 3, 11, 12, 13)}
#: The four molecules of the L-shaped cell (y, x in µm), each seen by every FOV whose footprint holds it.
NOTCH_MOLECULES = ((25.0, 18.25), (30.0, 18.25), (18.25, 22.0), (18.25, 27.0))


def _seen_by(points, genes, origins, shape):
    """Per FOV ``((z, y, x) index, gene)`` of each point (µm) whose position the FOV's footprint holds."""
    out = {fov: [] for fov in origins}
    for (y, x), gene in zip(points, genes):
        for fov, origin in origins.items():
            iy, ix = (y - origin[1]) / SPACING[1], (x - origin[2]) / SPACING[2]
            if -0.5 <= iy < shape[1] - 0.5 and -0.5 <= ix < shape[2] - 0.5:
                out[fov].append(((1.0, iy, ix), gene))
    return out


def fused_quad_notch() -> Scene:
    """``fused_quad`` with FOV4 a placeholder, so the covered region is L-shaped, and an L-shaped cell around the
    notch in place of the cells at the crossing, inside FOV4 and at the outer border."""
    shape = (72, 72)
    cells, nuclei = _discs(shape, NOTCH_DISCS), _discs(shape, NOTCH_DISCS, shrink=2)
    cells[34:67, 34:40] = NOTCH_CELL
    cells[34:40, 40:59] = NOTCH_CELL
    yy, xx = np.indices(shape)
    nuclei[(yy - 50) ** 2 + (xx - 36.5) ** 2 <= 4] = NOTCH_CELL
    molecules = quad_molecules(cells=NOTCH_DISCS)
    extra = _seen_by(NOTCH_MOLECULES, [GENES[m % 2] for m in range(len(NOTCH_MOLECULES))], QUAD_ORIGINS, (3, 40, 40))
    for fov, rows in extra.items():
        molecules[fov] = molecules[fov] + rows
    return Scene("fused_quad_notch", (3, 40, 40), dict(QUAD_ORIGINS), cells, nuclei, molecules,
                 statuses={"FOV4": "placeholder"})


# --- fused_pair_seams (W-429) -------------------------------------------------------------------------------------

#: The molecules of the two seam cases, name → ((y, x) µm at z 1 µm, gene); each seen by every FOV whose grid holds it.
SEAM_MOLECULES = {"m_a": ((3.0, 8.0), "A"), "m_b": ((3.0, 11.5), "A"), "m_c": ((3.0, 12.0), "B"),
                  "m_d": ((8.0, 2.0), "A"), "m_e": ((8.0, 10.0), "B"), "m_f": ((8.0, 17.0), "A"),
                  "m_g": ((11.0, 10.0), "B")}
SEAM_ORIGINS = {"FOV1": (0.0, 0.0, 0.0), "FOV2": (0.0, 0.0, 8.0)}


def seam_spot_ids() -> dict:
    """``{(fov_id, molecule name): spot_id}``: the row of each molecule in its FOV's goodSpots table, ``csv:<row>``."""
    out = {}
    for fov, origin in SEAM_ORIGINS.items():
        names = [n for n, ((y, x), _) in SEAM_MOLECULES.items() if -0.5 <= (x - origin[2]) / SPACING[2] < 23.5]
        out.update({(fov, name): f"csv:{row}" for row, name in enumerate(names)})
    return out


def fused_pair_seams() -> Scene:
    """The two seam cases on ``fused_pair``'s two FOVs: case high, cell 1, a disc of radius 5 pixels at (6, 19.5)
    pixels; case low, cell 2, the box of rows 15 to 18 and columns 2 to 37; the molecules ``m_a`` to ``m_g``. The
    page names no nuclei; each cell is its own nucleus, so that under the default exclusion no cell of either route
    is ``excluded_no_nucleus`` and no number of the page changes."""
    shape = (24, 40)
    yy, xx = np.indices(shape)
    cells = np.zeros(shape, np.uint32)
    cells[(yy - 6) ** 2 + (xx - 19.5) ** 2 <= 25] = 1
    cells[15:19, 2:38] = 2
    names = list(SEAM_MOLECULES)
    molecules = _seen_by([SEAM_MOLECULES[n][0] for n in names], [SEAM_MOLECULES[n][1] for n in names],
                         SEAM_ORIGINS, (3, 24, 24))
    return Scene("fused_pair_seams", (3, 24, 24), dict(SEAM_ORIGINS), cells, cells.copy(), molecules)


SCENES = {"fused_pair": fused_pair, "fused_quad": fused_quad, "fused_quad_notch": fused_quad_notch,
          "fused_pair_seams": fused_pair_seams}


# --- The scene as a dataset -----------------------------------------------------------------------------------

@dataclass
class SceneFolder:
    """A scene written as a dataset: the Dataset, its approved placement and FOV grids."""
    scene: Scene
    dataset: Dataset
    placement: object
    grids: dict
    extra: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return Path(self.dataset.output_root)

    @property
    def placement_dir(self) -> Path:
        return self.root / "stitching" / SAMPLE

    @property
    def export_dir(self) -> Path:
        return self.root / "export" / SAMPLE

    def image_path(self, name, fov) -> Path:
        return self.root / "images" / name / f"{fov}.tif"


def codebook():
    return Codebook(pd.DataFrame({"entry_id": ["111", "112"], "gene_id": list(GENES),
                                  "color_sequence": ["111", "112"]}),
                    round_labels=("round1", "round2", "round3"), channel_labels=("ch00", "ch01", "ch02", "ch03"))


def scene_placement(scene):
    table = positions_from_table(pd.DataFrame({"fov_id": scene.ids,
                                               "nominal_z": [o[0] for o in scene.origins.values()],
                                               "nominal_y": [o[1] for o in scene.origins.values()],
                                               "nominal_x": [o[2] for o in scene.origins.values()]}))
    result = placement_from_table(table, {f: scene.grid() for f in scene.ids}, sample=SAMPLE,
                                  origins=dict(scene.origins))
    if scene.statuses:
        from starfinder.stitching import PlacementResult
        fovs = result.fovs.assign(status=[scene.statuses.get(f, s) for f, s in zip(result.fovs.fov_id,
                                                                                   result.fovs.status)])
        result = PlacementResult(result.sample, fovs, result.links, result.record)
    return approve_placement(result, reviewer="test", note=scene.name)


def quad_grid_placement(folder) -> Path:
    """``fused_quad``'s placement with the grid cells ``ownership="midpoint_box"`` needs (W-446), saved under
    ``stitching_grid/<sample>/``: the same FOVs, grids and origins, from a ``positions_from_grid`` table of 2 × 2
    cells, column-major (FOV1 and FOV2 in column 0), step 15 µm = (1 − 0.25) × 40 × 0.5 µm, so each FOV's cell is
    its origin over the step rounded. The scene's own table is supplied and has no cells."""
    scene = folder.scene
    table = positions_from_grid(columns=2, rows=2, order="column_major_down_right", tile_shape_yx=scene.shape[1:],
                                spacing_yx=SPACING[1:], overlap=0.25, fov_ids=scene.ids)
    placement = approve_placement(placement_from_table(table, folder.grids, sample=SAMPLE,
                                                       origins=dict(scene.origins)),
                                  reviewer="test", note=f"{scene.name} with grid cells")
    path = folder.root / "stitching_grid" / SAMPLE
    save_placement(placement, path)
    return path


def write_scene(tmp_path, scene) -> SceneFolder:
    """The dataset of a scene under ``tmp_path``: placement, stains, reference images and molecule tables."""
    dataset = Dataset(tmp_path / "input", tmp_path / "output", DS, SAMPLE, "output", codebook=codebook())
    root = Path(dataset.output_root)
    placement = scene_placement(scene)
    save_placement(placement, root / "stitching" / SAMPLE)
    grids = {f: scene.grid() for f in scene.ids}
    for fov in scene.ids:
        save_volume(scene.cut(scene.stain("dapi"), fov), root / "images" / "ref_merged" / f"{fov}.tif",
                    metadata=grids[fov].metadata)
        for name in STAINS:
            path = root / "images" / name / f"{fov}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(path, scene.cut(scene.stain(name), fov), photometric="minisblack",
                             metadata={"axes": "ZYX"})
        rows = scene.molecules[fov]
        csv = root / "signal" / f"{fov}_goodSpots.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x": [p[2] + 1 for p, _ in rows], "y": [p[1] + 1 for p, _ in rows],
                      "z": [p[0] + 1 for p, _ in rows], "gene": [g for _, g in rows]}).to_csv(csv, index=False)
    return SceneFolder(scene, dataset, placement, grids)


def fusion_config(k, **changes):
    return FusionConfig(images=IMAGES, scale_yx=1.0 / k, **changes)


def fuse_scene(folder, k=1, **changes):
    """``fuse`` called alone on the scene's stains, as projections at ``scale_yx = 1/k``."""
    inputs = [FusionInput(f, folder.grids[f], {n: folder.image_path(n, f) for n in STAINS}) for f in folder.scene.ids]
    return fuse(folder.placement, inputs, config=fusion_config(k, **changes), output_dir=folder.export_dir)


def segmentation_grid(export_dir, image="dapi") -> ReferenceGrid:
    """The grid of a fused projection as the fusion section records it, frame ``sample:<sample>``."""
    record = json.loads((Path(export_dir) / "export.json").read_text())
    entry = next(o for o in record["fusion"]["outputs"] if o["image"] == image and o["format"] == "ome_tiff"
                 and o["dimensionality"] == "projection")
    metadata = ImageMetadata(f"sample:{record['sample']}", tuple(entry["spacing_zyx"]), tuple(entry["origin_zyx"]),
                             IDENTITY, "µm")
    return ReferenceGrid(tuple(entry["shape_zyx"]), metadata, "declared")


# --- truth_import, seeded_runs, length_configs, route_configs ---------------------------------------------------

def truth_labels(scene, grid, target="cell") -> np.ndarray:
    """The scene's labels on a grid: each pixel the scene label nearest its centre (0 outside the scene)."""
    labels = scene.cells if target == "cell" else scene.nuclei
    n_y, n_x = grid.shape_zyx[1:]
    (_, sy, sx), (_, oy, ox) = grid.metadata.spacing_zyx, grid.metadata.origin_zyx
    iy = scene_index(oy + np.arange(n_y) * sy, labels.shape[0])
    ix = scene_index(ox + np.arange(n_x) * sx, labels.shape[1])
    out = np.where((iy[:, None] >= 0) & (ix[None, :] >= 0), labels[iy.clip(0)[:, None], ix.clip(0)[None, :]], 0)
    return np.ascontiguousarray(out[np.newaxis].astype(np.uint32))


def truth_import(folder, grid, target="cell", *, path=None, labels=None) -> Path:
    """The truth labels on the grid written as a uint32 OME-TIFF with the grid's metadata, for an import run."""
    from starfinder.sample_output._tiff import write_ome_tiff
    path = Path(path or folder.root / "truth" / f"{target}.ome.tif")
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = truth_labels(folder.scene, grid, target) if labels is None else labels
    write_ome_tiff(path, labels[0], metadata=grid.metadata, annotation={"truth": target})
    return path


def truth_runs(cells_path, nuclei_path, **changes) -> SampleSegmentationConfig:
    """Both runs imported (``relabel: false``), so that ``cell_id`` is the truth ID."""
    runs = (SampleSegmentationRun("nucleus", "nucleus", (), LabelImportConfig(str(nuclei_path), "nucleus")),
            SampleSegmentationRun("cell", "cell", (), LabelImportConfig(str(cells_path), "cell")))
    return SampleSegmentationConfig(runs, "cell", "nucleus", **changes)


def seeded_runs(nuclei_path, **changes) -> SampleSegmentationConfig:
    """``nucleus`` imported from the truth nuclei; ``cell`` a seeded watershed (sigma 1 µm) on the fused flamingo."""
    runs = (SampleSegmentationRun("nucleus", "nucleus", (), LabelImportConfig(str(nuclei_path), "nucleus")),
            SampleSegmentationRun("cell", "cell", (("cytoplasm", "flamingo"),), SeededWatershedConfig(sigma_um=1.0),
                                  seeds="nucleus"))
    return SampleSegmentationConfig(runs, "cell", "nucleus", **changes)


def model_files(tmp_path):
    """A Cellpose model file and a StarDist model folder written by the test (never loaded)."""
    cellpose = Path(tmp_path) / "models" / "cpsam_test"
    cellpose.parent.mkdir(parents=True, exist_ok=True)
    cellpose.write_bytes(b"not a model")
    stardist = Path(tmp_path) / "models" / "stardist_test"
    stardist.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "thresholds.json", "weights_best.h5"):
        (stardist / name).write_text("{}")
    return cellpose, stardist


def length_configs(tmp_path) -> dict:
    cellpose, stardist = model_files(tmp_path)
    return {"cellpose": CellposeConfig(diameter=240, min_size=15, model_path=str(cellpose)),
            "stardist": StarDistConfig(scale=0.25, model_path=str(stardist)),
            "stardist_blocks": StarDistConfig(scale=1.0, block_size=(1, 256, 256), min_overlap=(0, 64, 64),
                                              context=(0, 52, 52), model_path=str(stardist)),
            "seeded_watershed": SeededWatershedConfig(sigma_um=1.5)}


def route_config(*, route=None, dimensionality="projection", projection="max", segment=True, fuse=True,
                 assemble=True, segmentation=None, k=1) -> ExportConfig:
    """One ``route_configs`` combination: the scene's stains fused at ``1/k``, the route and the segment settings."""
    return ExportConfig(assemble=assemble, fuse=fuse, route=route, segment=segment, segmentation=segmentation,
                        fusion=fusion_config(k, dimensionality=dimensionality, projection=projection))


# --- scene_plane (W-427) ----------------------------------------------------------------------------------------

def scene_plane_config():
    """The single-plane case of docs/synthetic-sample-scene.md on ``sample-tiny-v1`` (seed 42): FOVs of 1 × 36 × 36
    and jitter (0, 1.5, 1.5) voxels, everything else unchanged; no preset is added."""
    from dataclasses import replace

    from starfinder.synthetic import sample_scene_preset
    book, config = sample_scene_preset("sample-tiny-v1")
    return book, replace(config, acquisition=replace(config.acquisition, fov_shape_zyx=(1, 36, 36),
                                                     jitter_max_zyx=(0.0, 1.5, 1.5)),
                         fov=replace(config.fov, shape_zyx=(1, 36, 36)))


def scene_plane_small_config():
    """``scene_plane_small`` (W-429): the change of ``scene_plane`` applied to ``sample-small-v1`` (3 × 3 FOVs of
    1 × 96 × 96, jitter (0, 4.5, 4.5) voxels, seed 42), by ``dataclasses.replace``; no preset is added.

    Two further changes, because the generator refuses a one-plane scene with anything in Z: the background
    regions of the preset's appearance lie at z ≠ 0 of its 3D layout ("Z=1 requires region z=0"), so their z is set
    to 0, the rule the package's appearance applies to a one-plane shape (``synthetic._presets._appearance``); and
    the per-round translation bound (1.0, 2.0, 2.0) voxels becomes (0.0, 2.0, 2.0) ("Z=1 requires zero Z
    translation"), as ``sample-tiny-v1``'s is 0 in Z. Y and X are unchanged. The route after fusion imports the
    truth labels, so neither change reaches a compared value."""
    from dataclasses import replace

    from starfinder.synthetic import sample_scene_preset
    book, config = sample_scene_preset("sample-small-v1")
    background = config.fov.background
    background = replace(background, region_centers=tuple((0.0, y, x) for _, y, x in background.region_centers))
    geometry = config.fov.geometry
    geometry = replace(geometry, translation_max_zyx=(0.0, *geometry.translation_max_zyx[1:]))
    return book, replace(config, acquisition=replace(config.acquisition, fov_shape_zyx=(1, 96, 96),
                                                     jitter_max_zyx=(0.0, 4.5, 4.5)),
                         fov=replace(config.fov, shape_zyx=(1, 96, 96), background=background, geometry=geometry))


def write_scene_plane(tmp_path, name="scene_plane") -> SceneFolder:
    """``scene_plane`` (or ``scene_plane_small``) generated and written as a dataset: its truth placement
    (approved), its stain as both ``dapi`` and ``flamingo`` (float32 ZYX TIFFs), the reference images with each
    FOV's grid, the molecule tables of the saved scene as ``signal/`` goodSpots tables and the codebook.
    ``extra["scene"]`` is the generated scene."""
    from starfinder.synthetic import generate_sample_scene, save_sample_scene
    book, config = scene_plane_small_config() if name == "scene_plane_small" else scene_plane_config()
    generated = generate_sample_scene(book, config=config)
    saved = save_sample_scene(generated, Path(tmp_path) / "scene_plane")
    dataset = Dataset(tmp_path / "input", tmp_path / "output", DS, SAMPLE, "output", codebook=book)
    root = Path(dataset.output_root)
    truth = generated.placement.set_index("fov_id")
    ids = list(generated.positions.fov_id)
    grids = {f: ReferenceGrid(config.acquisition.fov_shape_zyx, generated.fov_metadata[f], "declared") for f in ids}
    origins = {f: tuple(float(v) for v in truth.loc[f, ["true_origin_z", "true_origin_y", "true_origin_x"]])
               for f in ids}
    table = positions_from_table(generated.positions)
    placement = approve_placement(placement_from_table(table, grids, sample=SAMPLE, origins=origins),
                                  reviewer="test", note="scene_plane truth placement")
    save_placement(placement, root / "stitching" / SAMPLE)
    for fov in ids:
        save_volume(generated.stains[fov], root / "images" / "ref_merged" / f"{fov}.tif",
                    metadata=grids[fov].metadata)
        for name in STAINS:
            path = root / "images" / name / f"{fov}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(path, np.asarray(generated.stains[fov], np.float32), photometric="minisblack",
                             metadata={"axes": "ZYX"})
        csv = root / "signal" / f"{fov}_goodSpots.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        csv.write_bytes((Path(saved) / "fovs" / fov / "molecules.csv").read_bytes())
    plane = PlaneScene(name, tuple(config.acquisition.fov_shape_zyx), origins, generated)
    return SceneFolder(plane, dataset, placement, grids, extra={"scene": generated, "book": book})


@dataclass
class PlaneScene:
    """``scene_plane`` as the helpers of this module read a scene: its FOVs, origins and the generated scene."""
    name: str
    shape: tuple
    origins: dict
    generated: object

    @property
    def ids(self):
        return list(self.origins)


def plane_truth_labels(scene: PlaneScene, grid, target="cell") -> np.ndarray:
    """``truth_import`` for a generated scene: each pixel of the grid takes the sample label grid's value nearest
    its centre (0 outside that grid)."""
    labels = scene.generated.sample_labels[target][0]
    record = scene.generated.record["sample_grid"]
    (_, sy, sx), (_, oy, ox) = record["spacing_zyx"], record["origin_zyx"]
    n_y, n_x = grid.shape_zyx[1:]
    (_, gy, gx), (_, py, px) = grid.metadata.spacing_zyx, grid.metadata.origin_zyx
    iy = np.floor((py + np.arange(n_y) * gy - oy) / sy + 0.5).astype(np.int64)
    ix = np.floor((px + np.arange(n_x) * gx - ox) / sx + 0.5).astype(np.int64)
    inside_y, inside_x = (iy >= 0) & (iy < labels.shape[0]), (ix >= 0) & (ix < labels.shape[1])
    out = np.where(inside_y[:, None] & inside_x[None, :], labels[iy.clip(0, labels.shape[0] - 1)[:, None],
                                                                 ix.clip(0, labels.shape[1] - 1)[None, :]], 0)
    return np.ascontiguousarray(out[np.newaxis].astype(np.uint32))


def scene_truth_labels(folder, grid, target="cell") -> np.ndarray:
    """The truth labels of any scene of this module on a grid."""
    if isinstance(folder.scene, PlaneScene):
        return plane_truth_labels(folder.scene, grid, target)
    return truth_labels(folder.scene, grid, target)


def fov_label_image(folder, fov, target="cell") -> np.ndarray:
    """A FOV's truth labels: the scene cut at its origin (hand-built), or the generated FOV label image."""
    if isinstance(folder.scene, PlaneScene):
        return np.ascontiguousarray(folder.scene.generated.fov_labels[fov][target], np.uint32)
    scene = folder.scene
    return np.ascontiguousarray(scene.cut(scene.cells if target == "cell" else scene.nuclei, fov), np.uint32)


def write_folder(tmp_path, name) -> SceneFolder:
    """A fixture of this module written as a dataset by its name."""
    if name in ("scene_plane", "scene_plane_small"):
        return write_scene_plane(tmp_path, name)
    return write_scene(tmp_path, SCENES[name]())


# --- per_fov_routes (W-427) ---------------------------------------------------------------------------------------

def per_fov_routes(folder, *, assignment=None, name="default"):
    """The route per FOV of the same fixture: per FOV the truth labels cut at its origin imported (``relabel``
    false), assigned with ``assignment`` (the AssignmentConfig of the route after fusion) and saved as the §2.9
    assignment folder ``checkpoints/<fov>/assignment/<name>/``. Returns the AssignmentResults by FOV."""
    import json as json_module

    from starfinder.assignment import AssignmentConfig, assign_molecules, molecule_table_from_csv
    from starfinder.assignment._persist import assignment_directory, write_assignment
    from starfinder.segmentation import import_labels
    assignment = AssignmentConfig() if assignment is None else assignment
    root = folder.root
    genes = tuple(folder.dataset.codebook.genes)
    results = {}
    for fov in folder.scene.ids:
        grid = folder.grids[fov]
        labels = {}
        for target in ("cell", "nucleus"):
            path = root / "truth_fov" / fov / f"{target}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(path, fov_label_image(folder, fov, target), photometric="minisblack",
                             metadata={"axes": "ZYX"})
            namespace = json_module.dumps([DS, SAMPLE, fov, None, target], separators=(",", ":"))
            labels[target] = import_labels(path, grid=grid, target=target, relabel=False, label_namespace=namespace)
        molecules = molecule_table_from_csv(root / "signal" / f"{fov}_goodSpots.csv",
                                            spot_namespace=json_module.dumps([DS, SAMPLE, fov, None],
                                                                             separators=(",", ":")), genes=genes)
        result = assign_molecules(molecules, labels["cell"], grid=grid, nuclei=labels["nucleus"], config=assignment)
        from dataclasses import replace
        result = replace(result, record=dict(result.record, name=name))
        write_assignment(assignment_directory(root / "checkpoints" / fov, name), result, metadata=grid.metadata)
        results[fov] = result
    return results
