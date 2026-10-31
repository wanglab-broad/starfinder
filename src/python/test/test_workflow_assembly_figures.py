"""The workflow draws the §2.9 assignment figures with the FOV ownership region (W-330): without an assembly block.

The sample of ``test_workflow_assembly_golden.py`` (three FOVs of 8×48×64 voxels, ZYX and YX label
images): the legacy reads_assignment adapter (``starfinder._legacy_workflow``, the MATLAB rule set's shared rule) runs
through a stub ``snakemake`` object, as that test runs it, and its figures carry no ownership. The figures with the
ownership region, which the export draws on the route of a saved assignment (``assignment.previews`` and
``export.previews``), moved to the assign and export commands' tests, test_workflow_export_tasks.py (W-750).
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.figure  # noqa: E402
import pytest  # noqa: E402

from .test_stitching_assembly_figure import EXCLUDED, scatters  # noqa: E402
from .test_workflow_assembly_golden import build_sample, run_adapter  # noqa: E402

pytestmark = [pytest.mark.workflow, pytest.mark.stitching]


def test_without_an_assembly_block_reads_assignment_draws_no_ownership(tmp_path, monkeypatch):
    """The legacy configuration: reads_assignment's figures carry no region line, no grey point and no stated count.
    (That no workflow file includes the old export rule is C8, test_workflow_boundary.py, W-750.)"""
    _, _, out = build_sample(tmp_path, "3d")
    saved = []
    real = matplotlib.figure.Figure.savefig
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig",
                        lambda figure, path, *a, **k: saved.append(figure) or real(figure, path, *a, **k))
    run_adapter(tmp_path, out, "Position001")
    assert len(saved) == 2
    for figure in saved:
        for ax in figure.axes[:2]:
            assert all(c.get_gid() != "ownership_region" for c in ax.collections)
            assert EXCLUDED not in scatters(ax) and ax.get_xlabel() == ""
