"""Guard the engineering comparison against hiding scientific or ungranted differences."""
from types import SimpleNamespace

import pytest

from . import workflow_engine_validation as E

pytestmark = [pytest.mark.workflow, pytest.mark.contract]


@pytest.mark.parametrize("reverse", [False, True])
def test_comparison_checks_engine_tokens_and_keeps_scientific_values(tmp_path, reverse):
    import json
    run = SimpleNamespace(work=tmp_path)
    left = {"run.json": json.dumps({"workflow": {"token": None, "wall_s": 1}, "counts": 2}).encode()}
    right = {"run.json": json.dumps({"workflow": {"token": "cache-token", "wall_s": 9}, "counts": 2}).encode()}
    def check(a, b):
        return E.compare(run, b if reverse else a, a if reverse else b,
                         left_engine="nextflow" if reverse else "snakemake",
                         right_engine="snakemake" if reverse else "nextflow", enforce=False)
    assert check(left, right) == []
    for token in (None, "", 1):
        invalid = {"run.json": json.dumps({"workflow": {"token": token}, "counts": 2}).encode()}
        with pytest.raises(AssertionError):
            check(left, invalid)
    with pytest.raises(AssertionError):
        check(right, right)
    right["run.json"] = json.dumps({"workflow": {"token": "cache-token"}, "counts": 3}).encode()
    assert check(left, right) == ["run.json"]


@pytest.mark.parametrize("reverse", [False, True])
def test_marker_is_only_snakemake_and_directly_inside_declared_directory(tmp_path, reverse):
    run = SimpleNamespace(work=tmp_path, data={"tasks": [{"id": "a", "outputs": ["folder/"], "record": "task.json"}]})
    def check(path, on_snakemake=True):
        sf, nf = ({path: b""}, {}) if on_snakemake else ({}, {path: b""})
        return E.compare(run, nf if reverse else sf, sf if reverse else nf, enforce=False,
                         left_engine="nextflow" if reverse else "snakemake",
                         right_engine="snakemake" if reverse else "nextflow")
    assert check("folder/.snakemake_timestamp") == []
    for path in ("folder/other", ".snakemake_timestamp", "folder/nested/.snakemake_timestamp"):
        assert check(path) == [path]
    assert check("folder/.snakemake_timestamp", False) == ["folder/.snakemake_timestamp"]


@pytest.mark.parametrize("reverse", [False, True])
def test_x11_hash_exceptions_apply_only_to_tasks_outside_rerun_set(tmp_path, reverse):
    import json
    run = SimpleNamespace(work=tmp_path, data={"tasks": [
        {"id": name, "record": name + ".json", "outputs": []} for name in ("cached", "rerun")]})
    def record(value):
        return json.dumps({"workflow": {"token": None, "configuration_sha256": value,
                                        "manifest_sha256": value, "fingerprint": "fixed"}}).encode()
    left = {n + ".json": record("old") for n in ("cached", "rerun")}
    right = {n + ".json": record("new") for n in ("cached", "rerun")}
    assert E.compare(run, right if reverse else left, left if reverse else right,
                     enforce=False, rerun_tasks={"rerun"}) == ["rerun.json"]
    assert E.compare(run, left, right, enforce=False) == ["cached.json", "rerun.json"]
    altered = json.loads(right["cached.json"])
    altered["workflow"]["fingerprint"] = "changed"
    right["cached.json"] = json.dumps(altered).encode()
    assert set(E.compare(run, right if reverse else left, left if reverse else right,
                         enforce=False, rerun_tasks={"rerun"})) == {"cached.json", "rerun.json"}


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("change", ["token", "configuration_sha256", "manifest_sha256", "formatting"])
def test_untouched_records_remain_byte_equal(tmp_path, reverse, change):
    import json
    before = json.dumps({"workflow": {"token": "cache", "configuration_sha256": "a", "manifest_sha256": "b"}}).encode()
    value = json.loads(before)
    if change != "formatting":
        value["workflow"][change] = "changed"
    after = json.dumps(value, indent=2).encode()
    E.assert_unchanged(before, before, "cached")
    with pytest.raises(AssertionError):
        E.assert_unchanged(after if reverse else before, before if reverse else after, "cached")


def test_comparison_keeps_float_bytes_and_does_not_omit_binary_or_html_files(tmp_path):
    run = SimpleNamespace(work=tmp_path)
    left = {"molecules.csv": b"gene_id,score\ngene,0.1\n", "reference.tif": b"pixels", "report.html": b"report"}
    for path, changed in [("molecules.csv", b"gene_id,score\ngene,0.10000000001\n"),
                          ("molecules.csv", b'gene_id,score\n"gene",0.1\n'),
                          ("reference.tif", b"changed"), ("report.html", b"changed")]:
        right = dict(left, **{path: changed})
        assert E.compare(run, left, right, enforce=False) == [path]


def test_snapshot_includes_files_declared_through_the_native_report_record(tmp_path):
    import json
    out = tmp_path / "output"
    folder = out / "report/sample"
    (folder / "tables").mkdir(parents=True)
    record = {"files": [{"path": "report.html"}, {"path": "tables/fovs.csv"}]}
    (folder / "report.json").write_text(json.dumps(record))
    (folder / "report.html").write_text("sample report")
    (folder / "tables/fovs.csv").write_text("fov_id,spots\nFOV_001,7\n")
    (out / "task.json").write_text("{}")
    run = SimpleNamespace(work=tmp_path, out=out, invocations=0,
                          data={"tasks": [{"kind": "report", "outputs": ["report/sample/report.json"],
                                           "record": "task.json"}]})
    result = E.snapshot(run)
    assert set(result) == {"task.json", "report/sample/report.json", "report/sample/report.html",
                           "report/sample/tables/fovs.csv"}
    assert result["report/sample/tables/fovs.csv"] == b"fov_id,spots\nFOV_001,7\n"


def _hash_page(value, extra="science"):
    return ('<div class="blk" id="b-files-configuration"><h3>Configuration</h3>'
            '<div class="panel"><table><tbody><tr><td>Workflow</td><td>workflow/resolved.json</td></tr>'
            f'<tr><td>Hash</td><td><code>{value}</code></td></tr>'
            f'</tbody></table></div></div><p>{extra}</p>').encode()


@pytest.mark.parametrize("reverse", [False, True])
def test_x11_report_hash_cell_is_scoped_and_checked_against_each_tree(tmp_path, reverse):
    import json
    run = SimpleNamespace(work=tmp_path, engine="snakemake", data={
        "membership": {"samples": {"sample": ["F1", "F2"]}}, "tasks": [
            {"id": "fov/F1", "kind": "fov", "record": "task.json", "outputs": ["fovs/F1/run.json"]},
            {"id": "report/sample", "kind": "report", "record": "report-task.json", "outputs": ["report/sample/report.html"]}]})
    page = "report/sample/report.html"
    def tree(value):
        return {page: _hash_page(value), "fovs/F1/run.json": json.dumps({"workflow_config": {"sha256": value}}).encode(),
                "fovs/F2/run.json": b'{"workflow_config":{"sha256":"not-the-first-FOV"}}'}
    left, right = tree("sha256:old"), tree("sha256:new")
    def compare(a, b, rerun_tasks={"report/sample"}):
        return E.compare(run, b if reverse else a, a if reverse else b,
                         enforce=False, rerun_tasks=rerun_tasks)
    assert compare(left, right) == []
    invalid = dict(right, **{page: _hash_page("sha256:wrong")})
    with pytest.raises(AssertionError, match="Hash cell differs"):
        compare(left, invalid)
    assert compare(left, right, {"fov/F1", "report/sample"}) == [page]
    assert compare(left, right, None) == [page]
    assert compare(left, dict(right, **{page: _hash_page("sha256:new", "changed science")})) == [page]
    for invalid_page in (_hash_page("sha256:new").replace(b'id="b-files-configuration"', b'data-id="b-files-configuration"'),
                         _hash_page("sha256:new").replace(b"<td>Hash</td>", b"<td>Other</td>"),
                         _hash_page("sha256:new") + _hash_page("sha256:new"),
                         _hash_page("sha256:new").replace(b'</tbody>', b'<tr><td>Hash</td><td><code>sha256:new</code></td></tr></tbody>')):
        with pytest.raises(AssertionError, match="expected one"):
            compare(left, dict(right, **{page: invalid_page}))


@pytest.mark.parametrize("reverse", [False, True])
def test_x11_file_names_include_unwalked_files_with_only_the_declared_marker_exception(tmp_path, reverse):
    run = SimpleNamespace(engine="snakemake", data={"tasks": [
        {"id": "export", "outputs": ["export/sample/current/"], "record": "task.json"}]})
    def compare(left, right, sf=True):
        return E.file_name_differences(run, right if reverse else left, left if reverse else right,
                                      left_engine="snakemake" if sf else "nextflow",
                                      right_engine="snakemake" if sf else "nextflow")
    common = {"export/sample/export.json", "report/sample/report.html"}
    assert compare(common | {"export/sample/current/.snakemake_timestamp"}, common) == []
    for path in ("export/sample/previews/FOV_001/assignment.png", "export/sample/previews/.snakemake_timestamp",
                 "export/sample/current/nested/.snakemake_timestamp", "report/foreign.txt"):
        assert compare(common | {path}, common) == [{"path": path, "side": "right" if reverse else "left"}]
    path = "export/sample/current/.snakemake_timestamp"
    assert compare(common | {path}, common, False) == [{"path": path, "side": "right" if reverse else "left"}]
