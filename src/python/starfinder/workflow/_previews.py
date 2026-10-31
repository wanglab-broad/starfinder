"""The previews of the assign and export commands (P5): options of their blocks, off by default.

``assignment.previews`` draws each FOV's assignment figures into ``fovs/<fov>/previews/``; ``export.previews`` draws
each assembled FOV's figures with its ownership into ``export/<sample>/previews/<fov>/`` on the route per FOV
(design/specs/orchestration.md, "Declared outputs"). The figures are those the legacy rules drew
(``_write_assignment_figures`` and ``_write_ownership_figures`` of the legacy adapters, which the Python workflow does
not import), over the FOV's reference image in place of the legacy DAPI image, which the Python workflow no longer
writes (D6). Each file is written to a temporary sibling and renamed into place (OP6).
"""
from __future__ import annotations

import os
from pathlib import Path

from starfinder import _layout

#: The figures of one FOV's assignment: the Z-maximum view, and the middle plane for a ZYX label image.
FIGURE = "assignment.png"
SINGLE_LAYER = "assignment_single_layer.png"


def assignment_figures(result, image_path, folder, *, ownership=None):
    """``<folder>/assignment.png`` and, for a ZYX label image, ``assignment_single_layer.png`` of an assignment.

    plot_assignment over the image ``image_path`` (left out when it is None or its Y, X are not those of the labels):
    the Z-maximum view and the middle plane; with ``ownership`` (a FOVOwnership), the FOV's owned region. Returns the
    written paths.
    """
    import matplotlib.pyplot as plt
    import tifffile
    from starfinder.assignment import plot_assignment
    image = None if image_path is None else tifffile.imread(image_path)
    image = image if image is not None and image.shape[-2:] == result.cell_labels.shape[1:] else None
    figures = {FIGURE: {}}
    n_z = result.cell_labels.shape[0]
    if n_z > 1:  # a ZYX label image; a YX one is a plane with the Z-maximum view only
        layer_image = image if image is None or image.ndim == 2 or image.shape[0] in (1, n_z) else None
        figures[SINGLE_LAYER] = {"view": "single_layer", "image": layer_image}
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, options in figures.items():
        figure = plot_assignment(result, **{"image": image, **options}, ownership=ownership)
        target = folder / name
        tmp = target.with_name(f".{os.urandom(8).hex()}.tmp.{name}")
        try:
            figure.savefig(tmp)
            os.replace(tmp, target)
        finally:
            plt.close(figure)
            tmp.unlink(missing_ok=True)
        written.append(target)
    return written


def ownership_figures(output_root, name, placement_dir, export_dir, declared, images):
    """The assignment figures of every assembled FOV whose saved assignment ``name`` is a declared input, with its
    ownership, into ``<export_dir>/previews/<fov>/``.

    The assembly is read from ``export_dir`` (load_assembly), the placement from ``placement_dir``, each FOV's
    assignment from ``fovs/<fov>/assignment/<name>/``; ``declared`` are the task's declared input paths and
    ``images`` maps a FOV to the image the figures are drawn over (None: no image).
    """
    from starfinder.assembly import fov_ownership, load_assembly
    from starfinder.assignment._persist import RECORD_FILE, assignment_directory, read_assignment
    from starfinder.stitching import load_placement
    declared = {os.path.abspath(p) for p in declared}
    assembly, placement = load_assembly(export_dir), load_placement(placement_dir)
    fovs = assembly.fovs
    written = []
    for fov_id in fovs["fov_id"][fovs["kind"].eq("assignment").fillna(False).to_numpy(dtype=bool)]:
        folder = assignment_directory(_layout.fov_dir(output_root, fov_id), name)
        if os.path.abspath(folder / RECORD_FILE) not in declared:
            continue
        result = read_assignment(folder, name)
        written += assignment_figures(result, images.get(fov_id), Path(export_dir) / _layout.PREVIEWS / fov_id,
                                      ownership=fov_ownership(assembly, fov_id, placement=placement))
    return written
